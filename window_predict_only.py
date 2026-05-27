# -*- coding: utf-8 -*-
"""
window_predict_only.py

窗口级 FCU 故障分类预测脚本。

前置条件：
1. 已用 window_data_preprocess.py 生成测试集窗口特征：
   artifacts_window_data/window_test_features.csv
   artifacts_window_data/window_test_file_summary.csv

2. 已用 window_train_only.py 训练得到窗口级模型：
   artifacts_window_train/best_window_pipeline.pkl

功能：
1. 加载窗口级训练模型；
2. 读取测试集窗口特征；
3. 对每个窗口输出类别概率；
4. 按原始文件 group_id 聚合窗口概率，得到文件级预测类别；
5. 输出窗口级预测结果、文件级详细预测结果和最终提交文件。

单模型预测示例：
python window_predict_only.py ^
  --data_dir "artifacts_window_data" ^
  --model_files "artifacts_window_train\\best_window_pipeline.pkl" ^
  --output_file "test_results\\window_file_predictions.csv" ^
  --submission_file "test_results\\window_submission.csv" ^
  --window_output_file "test_results\\window_level_predictions.csv"

多模型概率集成预测示例：
python window_predict_only.py ^
  --data_dir "artifacts_window_data" ^
  --model_files "artifacts_window_train\\best_window_pipeline.pkl,artifacts_window_train2\\best_window_pipeline.pkl" ^
  --output_file "test_results\\window_ensemble_file_predictions.csv" ^
  --submission_file "test_results\\window_ensemble_submission.csv" ^
  --window_output_file "test_results\\window_ensemble_level_predictions.csv"

说明：
- 多模型预测时，脚本会对各模型的窗口级概率做平均，再按文件聚合。
- 最终提交文件默认只包含 filename 和 predicted_label 两列。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 基础工具
# ============================================================

def parse_model_files(s: str) -> List[Path]:
    files = [Path(x.strip()) for x in s.split(",") if x.strip()]
    if not files:
        raise ValueError("请至少提供一个模型文件。")

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"模型文件不存在：{f}")

    return files


def parse_weights(s: str | None, n: int) -> np.ndarray:
    """
    解析多模型权重。
    未指定则等权平均。
    """
    if s is None or not s.strip():
        return np.ones(n, dtype="float64") / n

    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"权重数量 {len(vals)} 与模型数量 {n} 不一致。")

    w = np.asarray(vals, dtype="float64")
    if np.any(w < 0):
        raise ValueError("模型权重不能为负数。")
    if w.sum() <= 0:
        raise ValueError("模型权重之和必须大于 0。")

    return w / w.sum()


def load_window_test_data(data_dir: Path, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    读取窗口级测试特征，并按训练阶段 feature_names 对齐。
    """
    test_file = data_dir / "window_test_features.csv"
    if not test_file.exists():
        raise FileNotFoundError(
            f"找不到测试窗口特征表：{test_file}\n"
            f"请先运行 window_data_preprocess.py test 生成测试窗口特征。"
        )

    df = pd.read_csv(test_file)

    for c in ["group_id", "source_file"]:
        if c not in df.columns:
            raise KeyError(f"测试窗口特征表缺少必要列：{c}")

    X = df.reindex(columns=feature_names)
    groups = df["group_id"].astype(str).to_numpy()
    source_files = df["source_file"].astype(str).to_numpy()

    return X, groups, source_files, df


def align_proba_to_classes(
    proba: np.ndarray,
    model_classes: List,
    global_classes: List,
) -> np.ndarray:
    """
    将某个模型的概率矩阵对齐到统一类别顺序。
    """
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype="float64")
    class_to_idx = {c: i for i, c in enumerate(model_classes)}

    for j, c in enumerate(global_classes):
        if c not in class_to_idx:
            raise ValueError(f"模型概率中缺少类别 {c}，无法与全局类别对齐。")
        aligned[:, j] = proba[:, class_to_idx[c]]

    return aligned


# ============================================================
# 2. 文件级聚合
# ============================================================

def aggregate_window_proba_to_file(
    proba: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    classes: List,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    将窗口级概率聚合为文件级概率。

    aggregation:
    - mean: 对同一文件的所有窗口概率取平均，推荐
    - median: 对窗口概率取中位数，较稳健
    - max: 对每类概率取最大值，适合故障局部明显但可能更激进
    """
    prob_cols = [f"prob_class_{c}" for c in classes]

    df = pd.DataFrame(proba, columns=prob_cols)
    df["group_id"] = groups
    df["source_file"] = source_files

    if aggregation == "mean":
        agg_func = "mean"
    elif aggregation == "median":
        agg_func = "median"
    elif aggregation == "max":
        agg_func = "max"
    else:
        raise ValueError(f"未知聚合方式：{aggregation}")

    agg_dict = {c: agg_func for c in prob_cols}
    agg_dict["source_file"] = "first"

    out = df.groupby("group_id", sort=False).agg(agg_dict).reset_index()

    prob_mat = out[prob_cols].to_numpy()
    pred_idx = np.argmax(prob_mat, axis=1)

    out["predicted_label"] = [classes[i] for i in pred_idx]
    out["top1_probability"] = prob_mat[np.arange(len(out)), pred_idx]

    if len(classes) >= 2:
        sorted_idx = np.argsort(prob_mat, axis=1)[:, ::-1]
        top2_idx = sorted_idx[:, 1]
        out["top2_label"] = [classes[i] for i in top2_idx]
        out["top2_probability"] = prob_mat[np.arange(len(out)), top2_idx]
        out["prob_margin"] = out["top1_probability"] - out["top2_probability"]

    # 调整列顺序
    front_cols = [
        "source_file",
        "group_id",
        "predicted_label",
        "top1_probability",
        "top2_label",
        "top2_probability",
        "prob_margin",
    ]
    exist_front = [c for c in front_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in exist_front]
    out = out[exist_front + other_cols]

    return out


# ============================================================
# 3. 单模型预测
# ============================================================

def predict_one_model(
    model_file: Path,
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, Dict]:
    """
    使用一个窗口模型预测测试窗口概率。

    返回：
    proba, groups, source_files, classes, model_info
    """
    print("\n========== 加载窗口模型 ==========")
    print(f"模型文件: {model_file}")

    artifact = joblib.load(model_file)

    if "pipeline" not in artifact:
        raise KeyError(f"{model_file} 中没有 pipeline 字段。")

    pipe = artifact["pipeline"]
    feature_names = artifact.get("feature_names", None)
    class_labels = artifact.get("class_labels", None)

    if feature_names is None:
        raise KeyError(f"{model_file} 中没有 feature_names 字段，无法对齐测试特征。")
    if class_labels is None:
        raise KeyError(f"{model_file} 中没有 class_labels 字段，无法确定类别顺序。")

    X_test, groups, source_files, raw_df = load_window_test_data(data_dir, feature_names)

    print(f"模型名称: {artifact.get('model_name', 'unknown')}")
    print(f"模型 k: {artifact.get('k', 'unknown')}")
    print(f"训练 CV macro-F1: {artifact.get('best_macro_f1', 'unknown')}")
    print(f"测试窗口数: {X_test.shape[0]}")
    print(f"测试文件数: {len(np.unique(groups))}")
    print(f"特征维度: {X_test.shape[1]}")
    print(f"缺失值比例均值: {float(X_test.isna().mean().mean()):.6f}")

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        model_classes = list(pipe.classes_)
    else:
        print("警告：模型不支持 predict_proba，将使用 predict 结果构造 one-hot 概率。")
        pred = pipe.predict(X_test)
        model_classes = list(class_labels)
        proba = np.zeros((len(pred), len(model_classes)), dtype="float64")
        c_to_i = {c: i for i, c in enumerate(model_classes)}
        for i, p in enumerate(pred):
            proba[i, c_to_i[p]] = 1.0

    global_classes = list(class_labels)
    proba = align_proba_to_classes(proba, model_classes, global_classes)

    model_info = {
        "model_file": str(model_file),
        "model_name": artifact.get("model_name", "unknown"),
        "k": artifact.get("k", None),
        "best_macro_f1": artifact.get("best_macro_f1", None),
        "n_features": len(feature_names),
        "aggregation_in_training": artifact.get("aggregation", None),
    }

    return proba, groups, source_files, global_classes, model_info


# ============================================================
# 4. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="窗口级 FCU 测试集预测脚本")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="artifacts_window_data",
        help="窗口数据目录，需包含 window_test_features.csv"
    )

    parser.add_argument(
        "--model_files",
        type=str,
        required=True,
        help="一个或多个窗口模型 pkl 文件，用英文逗号分隔"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="多模型集成权重，例如 0.5,0.3,0.2；不填则等权平均"
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "max"],
        help="窗口概率聚合为文件概率的方式，默认 mean"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="test_results/window_file_predictions.csv",
        help="文件级详细预测结果"
    )

    parser.add_argument(
        "--submission_file",
        type=str,
        default="test_results/window_submission.csv",
        help="最终提交文件，仅包含 filename 和 predicted_label"
    )

    parser.add_argument(
        "--window_output_file",
        type=str,
        default="test_results/window_level_predictions.csv",
        help="窗口级预测结果"
    )

    parser.add_argument(
        "--info_file",
        type=str,
        default="test_results/window_prediction_info.json",
        help="预测配置信息"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"窗口数据目录不存在：{data_dir}")

    model_files = parse_model_files(args.model_files)
    weights = parse_weights(args.weights, len(model_files))

    print("\n========== 窗口级测试集预测 ==========")
    print(f"窗口数据目录: {data_dir}")
    print(f"模型数量: {len(model_files)}")
    print(f"模型权重: {weights.tolist()}")
    print(f"文件级聚合方式: {args.aggregation}")

    weighted_probas = []
    groups_ref = None
    source_files_ref = None
    classes_ref = None
    model_infos = []

    for i, model_file in enumerate(model_files):
        proba, groups, source_files, classes, info = predict_one_model(
            model_file=model_file,
            data_dir=data_dir,
        )

        if groups_ref is None:
            groups_ref = groups
            source_files_ref = source_files
            classes_ref = classes
        else:
            if not np.array_equal(groups, groups_ref):
                raise ValueError(f"模型 {model_file} 读取到的 group_id 顺序与第一个模型不一致。")
            if not np.array_equal(source_files, source_files_ref):
                raise ValueError(f"模型 {model_file} 读取到的 source_file 顺序与第一个模型不一致。")
            if list(classes) != list(classes_ref):
                raise ValueError(f"模型 {model_file} 的类别顺序与第一个模型不一致。")

        weighted_probas.append(proba * weights[i])
        model_infos.append(info)

    mean_proba = np.sum(weighted_probas, axis=0)

    # 窗口级预测表
    prob_cols = [f"prob_class_{c}" for c in classes_ref]
    win_pred = pd.DataFrame(mean_proba, columns=prob_cols)
    win_pred["group_id"] = groups_ref
    win_pred["source_file"] = source_files_ref
    win_idx = np.argmax(mean_proba, axis=1)
    win_pred["window_predicted_label"] = [classes_ref[i] for i in win_idx]
    win_pred["window_top1_probability"] = mean_proba[np.arange(len(win_idx)), win_idx]

    # 文件级预测表
    file_pred = aggregate_window_proba_to_file(
        proba=mean_proba,
        groups=groups_ref,
        source_files=source_files_ref,
        classes=classes_ref,
        aggregation=args.aggregation,
    )

    output_file = Path(args.output_file)
    submission_file = Path(args.submission_file)
    window_output_file = Path(args.window_output_file)
    info_file = Path(args.info_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_file.parent.mkdir(parents=True, exist_ok=True)
    window_output_file.parent.mkdir(parents=True, exist_ok=True)
    info_file.parent.mkdir(parents=True, exist_ok=True)

    file_pred.to_csv(output_file, index=False, encoding="utf-8-sig")
    win_pred.to_csv(window_output_file, index=False, encoding="utf-8-sig")

    submission = file_pred[["source_file", "predicted_label"]].copy()
    submission = submission.rename(columns={"source_file": "filename"})
    submission.to_csv(submission_file, index=False, encoding="utf-8-sig")

    info = {
        "data_dir": str(data_dir),
        "model_files": [str(p) for p in model_files],
        "weights": weights.tolist(),
        "aggregation": args.aggregation,
        "classes": [str(c) for c in classes_ref],
        "n_window_samples": int(len(win_pred)),
        "n_files": int(len(file_pred)),
        "output_file": str(output_file),
        "submission_file": str(submission_file),
        "window_output_file": str(window_output_file),
        "model_infos": model_infos,
        "predicted_label_distribution": {
            str(k): int(v) for k, v in file_pred["predicted_label"].value_counts().sort_index().items()
        },
        "low_confidence_count_top1_lt_0_6": int((file_pred["top1_probability"] < 0.6).sum()),
        "low_margin_count_lt_0_1": int((file_pred["prob_margin"] < 0.1).sum()) if "prob_margin" in file_pred.columns else None,
    }

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 预测完成 ==========")
    print(f"测试窗口数: {len(win_pred)}")
    print(f"测试文件数: {len(file_pred)}")
    print(f"文件级预测结果: {output_file}")
    print(f"窗口级预测结果: {window_output_file}")
    print(f"提交文件: {submission_file}")
    print(f"预测信息: {info_file}")

    print("\n预测类别分布:")
    print(file_pred["predicted_label"].value_counts().sort_index())

    print("\n低置信度统计:")
    print(f"top1_probability < 0.6: {(file_pred['top1_probability'] < 0.6).sum()}")
    if "prob_margin" in file_pred.columns:
        print(f"prob_margin < 0.1: {(file_pred['prob_margin'] < 0.1).sum()}")

    print("\n前 10 行文件级预测结果:")
    print(file_pred.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
