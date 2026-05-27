# -*- coding: utf-8 -*-
"""
window_train_models.py

窗口级 FCU 多模型训练与预测脚本。

前置步骤：
先使用 window_data_preprocess.py 生成：
- artifacts_window_data/window_train_features.csv
- artifacts_window_data/window_test_features.csv
- artifacts_window_data/window_schema.json

核心思想：
1. 训练阶段以“窗口”为训练样本；
2. 交叉验证阶段按 group_id=原始文件 分组，保证同一文件切出的窗口不会同时出现在训练集和验证集；
3. 验证阶段先预测窗口概率，再按原始文件聚合为文件级概率；
4. 最终评价指标仍然是文件级 macro-F1，与测试集任务一致；
5. 支持多模型对比：ExtraTrees、RandomForest、HistGradientBoosting，以及可选 LightGBM、XGBoost、CatBoost。

训练示例：
python window_train_models.py train ^
  --data_dir "artifacts_window_data" ^
  --output_dir "artifacts_window_models" ^
  --k_list 300,400,500 ^
  --cv 5

预测示例：
python window_train_models.py predict ^
  --data_dir "artifacts_window_data" ^
  --model_file "artifacts_window_models\\best_window_model.pkl" ^
  --output_file "test_results\\window_pred_detail.csv" ^
  --submission_file "test_results\\window_submission.csv"
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. 工具函数
# ============================================================

def parse_int_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return sorted(set(vals))


def load_schema(data_dir: Path) -> dict:
    schema_file = data_dir / "window_schema.json"
    if not schema_file.exists():
        raise FileNotFoundError(f"找不到 schema 文件：{schema_file}")
    with open(schema_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_window_train_data(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """
    读取窗口级训练特征。

    返回：
    X: 窗口级特征
    y: 窗口级标签
    groups: group_id，原始文件 ID
    source_files: 原始文件名
    feature_names: 特征列名
    schema: schema
    """
    schema = load_schema(data_dir)

    train_file = data_dir / "window_train_features.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"找不到窗口训练特征表：{train_file}")

    df = pd.read_csv(train_file)

    label_col = schema.get("label_col", "label")
    group_col = schema.get("group_col", "group_id")
    feature_names = schema["feature_names"]

    required_cols = [label_col, group_col, "source_file"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"训练特征表缺少必要列：{c}")

    # 对齐特征列
    X = df.reindex(columns=feature_names)
    y = df[label_col].to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    source_files = df["source_file"].astype(str).to_numpy()

    return X, y, groups, source_files, feature_names, schema


def load_window_test_data(data_dir: Path, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    读取窗口级测试特征。
    """
    test_file = data_dir / "window_test_features.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"找不到窗口测试特征表：{test_file}")

    df = pd.read_csv(test_file)

    for c in ["group_id", "source_file"]:
        if c not in df.columns:
            raise KeyError(f"测试特征表缺少必要列：{c}")

    X = df.reindex(columns=feature_names)
    groups = df["group_id"].astype(str).to_numpy()
    source_files = df["source_file"].astype(str).to_numpy()

    return X, groups, source_files


def get_group_labels(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从窗口级标签恢复文件级标签。
    同一个 group 内的标签应完全一致。
    返回 unique_groups, group_labels
    """
    df = pd.DataFrame({"group": groups, "y": y})
    rows = []
    for g, sub in df.groupby("group", sort=False):
        labels = sub["y"].unique()
        if len(labels) != 1:
            raise ValueError(f"group {g} 内出现多个标签：{labels}")
        rows.append((g, labels[0]))
    unique_groups = np.array([r[0] for r in rows])
    group_labels = np.array([r[1] for r in rows])
    return unique_groups, group_labels


def make_group_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int = 42,
):
    """
    生成按文件分组的交叉验证划分。

    优先使用 StratifiedGroupKFold；
    如果当前 sklearn 版本不支持，则使用 GroupKFold。
    """
    unique_groups, group_labels = get_group_labels(y, groups)
    min_count = pd.Series(group_labels).value_counts().min()
    n_splits = min(n_splits, int(min_count))

    if n_splits < 2:
        raise ValueError("最小类别文件数不足 2，无法进行交叉验证。")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splitter = cv.split(np.zeros(len(y)), y, groups)
        cv_name = "StratifiedGroupKFold"
    except Exception:
        cv = GroupKFold(n_splits=n_splits)
        splitter = cv.split(np.zeros(len(y)), y, groups)
        cv_name = "GroupKFold"

    return list(splitter), cv_name, n_splits


def aggregate_window_proba_to_file(
    proba: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    classes: List,
    y_window: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    将窗口级概率按 group_id 聚合为文件级概率。
    默认采用概率均值。
    """
    prob_cols = [f"prob_class_{c}" for c in classes]
    df = pd.DataFrame(proba, columns=prob_cols)
    df["group_id"] = groups
    df["source_file"] = source_files

    if y_window is not None:
        df["label"] = y_window

    agg_dict = {c: "mean" for c in prob_cols}
    agg_dict["source_file"] = "first"

    if y_window is not None:
        agg_dict["label"] = "first"

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

    return out


# ============================================================
# 2. 模型构建
# ============================================================

def build_models(random_state: int = 42) -> Dict[str, object]:
    """
    构建候选模型。
    """
    models = {}

    models["ExtraTrees_regularized"] = ExtraTreesClassifier(
        n_estimators=1200,
        max_features=0.5,
        min_samples_leaf=2,
        min_samples_split=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    models["ExtraTrees_deep"] = ExtraTreesClassifier(
        n_estimators=1200,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=random_state,
    )

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        print("提示：未安装 lightgbm，跳过 LightGBM。")

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        print("提示：未安装 xgboost，跳过 XGBoost。")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.03,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
        )
    except Exception:
        print("提示：未安装 catboost，跳过 CatBoost。")

    return models


def build_pipeline(model, k: int, random_state: int = 42) -> Pipeline:
    """
    构建窗口级训练 Pipeline。
    """
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
        k=k,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", selector),
        ("clf", model),
    ])

    return pipe


# ============================================================
# 3. 交叉验证：窗口训练，文件级评价
# ============================================================

def evaluate_one_model(
    model_name: str,
    model,
    k: int,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    splits,
    random_state: int,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    对单个模型 + k 做 Group CV。
    """
    fold_rows = []
    all_file_pred = []

    classes_global = sorted(np.unique(y).tolist())

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        pipe = build_pipeline(clone(model), k=k, random_state=random_state)

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y[valid_idx]
        groups_valid = groups[valid_idx]
        source_valid = source_files[valid_idx]

        pipe.fit(X_train, y_train)

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_valid)
            classes = list(pipe.classes_)
        else:
            pred = pipe.predict(X_valid)
            classes = classes_global
            proba = np.zeros((len(pred), len(classes)), dtype="float64")
            c_to_i = {c: i for i, c in enumerate(classes)}
            for i, p in enumerate(pred):
                proba[i, c_to_i[p]] = 1.0

        # 对齐类别顺序到 classes_global
        aligned = np.zeros((proba.shape[0], len(classes_global)), dtype="float64")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for j, c in enumerate(classes_global):
            if c in class_to_idx:
                aligned[:, j] = proba[:, class_to_idx[c]]

        file_pred = aggregate_window_proba_to_file(
            proba=aligned,
            groups=groups_valid,
            source_files=source_valid,
            classes=classes_global,
            y_window=y_valid,
        )

        y_true_file = file_pred["label"].to_numpy()
        y_pred_file = file_pred["predicted_label"].to_numpy()

        acc = accuracy_score(y_true_file, y_pred_file)
        macro = f1_score(y_true_file, y_pred_file, average="macro")
        weighted = f1_score(y_true_file, y_pred_file, average="weighted")

        fold_rows.append({
            "fold": fold,
            "n_valid_files": int(len(file_pred)),
            "n_valid_windows": int(len(valid_idx)),
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
        })

        file_pred["fold"] = fold
        file_pred["model"] = model_name
        file_pred["k"] = k
        all_file_pred.append(file_pred)

    fold_df = pd.DataFrame(fold_rows)

    result = {
        "model": model_name,
        "k": int(k),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(fold_df["macro_f1"].mean()),
        "macro_f1_std": float(fold_df["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
        "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=0)),
    }

    if output_dir is not None:
        pred_df = pd.concat(all_file_pred, ignore_index=True)
        pred_path = output_dir / f"cv_file_predictions_{model_name}_k{k}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return result


# ============================================================
# 4. 训练主流程
# ============================================================

def train_main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, source_files, feature_names, schema = load_window_train_data(data_dir)

    print("\n========== 窗口训练数据检查 ==========")
    print(f"窗口样本数: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    print(f"文件 group 数: {len(np.unique(groups))}")
    print("文件级类别分布:", dict(pd.Series(get_group_labels(y, groups)[1]).value_counts().sort_index()))
    print("窗口级类别分布:", dict(pd.Series(y).value_counts().sort_index()))
    print(f"缺失值比例均值: {float(X.isna().mean().mean()):.6f}")

    splits, cv_name, n_splits = make_group_cv_splits(
        y=y,
        groups=groups,
        n_splits=args.cv,
        random_state=args.random_state,
    )

    print("\n========== 分组交叉验证设置 ==========")
    print(f"CV 方法: {cv_name}")
    print(f"折数: {n_splits}")
    print("说明：同一原始文件的窗口不会同时出现在训练集和验证集。")

    models = build_models(random_state=args.random_state)
    k_list = parse_int_list(args.k_list)

    results = []
    best_score = -np.inf
    best_model_name = None
    best_k = None
    best_model = None

    print("\n========== 开始窗口级多模型训练，文件级评估 ==========")

    for k in k_list:
        for model_name, model in models.items():
            exp = f"{model_name}_k{k}"
            print(f"\n正在评估：{exp}")

            try:
                res = evaluate_one_model(
                    model_name=model_name,
                    model=model,
                    k=k,
                    X=X,
                    y=y,
                    groups=groups,
                    source_files=source_files,
                    splits=splits,
                    random_state=args.random_state,
                    output_dir=output_dir if args.save_fold_predictions else None,
                )

                print(
                    f"  accuracy={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}, "
                    f"macro-F1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}"
                )

                results.append(res)

                if res["macro_f1_mean"] > best_score:
                    best_score = res["macro_f1_mean"]
                    best_model_name = model_name
                    best_k = k
                    best_model = clone(model)
                    print("  >>> 当前最优模型已更新")

            except Exception as e:
                print(f"  实验失败，已跳过：{e}")
                results.append({
                    "model": model_name,
                    "k": k,
                    "error": str(e),
                })

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1_mean",
        ascending=False,
        na_position="last",
    )

    cv_path = output_dir / "window_cv_results.csv"
    results_df.to_csv(cv_path, index=False, encoding="utf-8-sig")

    print("\n========== 窗口模型交叉验证 Top 10 ==========")
    print(results_df.head(10).to_string(index=False))

    if best_model is None:
        raise RuntimeError("所有窗口模型实验均失败，请检查数据和依赖。")

    print("\n========== 使用全体窗口样本训练最终模型 ==========")
    print(f"最优模型: {best_model_name}")
    print(f"最优 k: {best_k}")
    print(f"最优文件级 macro-F1: {best_score:.4f}")

    best_pipe = build_pipeline(best_model, k=best_k, random_state=args.random_state)
    best_pipe.fit(X, y)

    model_path = output_dir / "best_window_model.pkl"

    artifact = {
        "pipeline": best_pipe,
        "model_name": best_model_name,
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "feature_names": feature_names,
        "class_labels": sorted(np.unique(y).tolist()),
        "schema": schema,
        "cv_results": results,
        "aggregation": "mean_proba_by_group",
        "group_col": "group_id",
        "source_file_col": "source_file",
    }

    joblib.dump(artifact, model_path)

    info = {
        "model_name": best_model_name,
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "n_window_samples": int(X.shape[0]),
        "n_groups": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "cv_method": cv_name,
        "n_splits": int(n_splits),
    }

    info_path = output_dir / "window_model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 最终模型保存完成 ==========")
    print(f"模型文件: {model_path}")
    print(f"交叉验证结果: {cv_path}")
    print(f"模型信息: {info_path}")


# ============================================================
# 5. 预测主流程
# ============================================================

def predict_main(args):
    data_dir = Path(args.data_dir)
    model_file = Path(args.model_file)

    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_file}")

    artifact = joblib.load(model_file)
    pipe = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    classes_global = artifact["class_labels"]

    X_test, groups, source_files = load_window_test_data(data_dir, feature_names)

    print("\n========== 窗口测试数据检查 ==========")
    print(f"测试窗口数: {X_test.shape[0]}")
    print(f"测试文件数: {len(np.unique(groups))}")
    print(f"特征维度: {X_test.shape[1]}")
    print(f"模型: {artifact.get('model_name')}")
    print(f"k: {artifact.get('k')}")
    print(f"训练 CV macro-F1: {artifact.get('best_macro_f1')}")

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        classes = list(pipe.classes_)
    else:
        pred = pipe.predict(X_test)
        classes = classes_global
        proba = np.zeros((len(pred), len(classes)), dtype="float64")
        c_to_i = {c: i for i, c in enumerate(classes)}
        for i, p in enumerate(pred):
            proba[i, c_to_i[p]] = 1.0

    # 对齐类别顺序
    aligned = np.zeros((proba.shape[0], len(classes_global)), dtype="float64")
    c_to_i = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(classes_global):
        if c in c_to_i:
            aligned[:, j] = proba[:, c_to_i[c]]

    # 保存窗口级预测
    prob_cols = [f"prob_class_{c}" for c in classes_global]
    win_pred = pd.DataFrame(aligned, columns=prob_cols)
    win_pred["group_id"] = groups
    win_pred["source_file"] = source_files
    win_pred["window_predicted_label"] = [classes_global[i] for i in np.argmax(aligned, axis=1)]
    win_pred["window_top1_probability"] = np.max(aligned, axis=1)

    file_pred = aggregate_window_proba_to_file(
        proba=aligned,
        groups=groups,
        source_files=source_files,
        classes=classes_global,
        y_window=None,
    )

    output_file = Path(args.output_file)
    submission_file = Path(args.submission_file)
    window_output_file = Path(args.window_output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_file.parent.mkdir(parents=True, exist_ok=True)
    window_output_file.parent.mkdir(parents=True, exist_ok=True)

    file_pred.to_csv(output_file, index=False, encoding="utf-8-sig")
    win_pred.to_csv(window_output_file, index=False, encoding="utf-8-sig")

    submission = file_pred[["source_file", "predicted_label"]].copy()
    submission = submission.rename(columns={"source_file": "filename"})
    submission.to_csv(submission_file, index=False, encoding="utf-8-sig")

    print("\n========== 测试集预测完成 ==========")
    print(f"文件级预测结果: {output_file}")
    print(f"窗口级预测结果: {window_output_file}")
    print(f"提交文件: {submission_file}")

    print("\n预测类别分布:")
    print(file_pred["predicted_label"].value_counts().sort_index())

    print("\n前 10 行文件级预测:")
    print(file_pred.head(10).to_string(index=False))


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="窗口级 FCU 多模型训练与预测脚本")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train", help="训练窗口级多模型")
    p_train.add_argument("--data_dir", type=str, default="artifacts_window_data")
    p_train.add_argument("--output_dir", type=str, default="artifacts_window_models")
    p_train.add_argument("--k_list", type=str, default="300,400,500")
    p_train.add_argument("--cv", type=int, default=5)
    p_train.add_argument("--random_state", type=int, default=42)
    p_train.add_argument("--save_fold_predictions", action="store_true")

    p_predict = subparsers.add_parser("predict", help="使用窗口模型预测测试集")
    p_predict.add_argument("--data_dir", type=str, default="artifacts_window_data")
    p_predict.add_argument("--model_file", type=str, required=True)
    p_predict.add_argument("--output_file", type=str, default="test_results/window_file_predictions.csv")
    p_predict.add_argument("--submission_file", type=str, default="test_results/window_submission.csv")
    p_predict.add_argument("--window_output_file", type=str, default="test_results/window_level_predictions.csv")

    args = parser.parse_args()

    if args.mode == "train":
        train_main(args)
    elif args.mode == "predict":
        predict_main(args)


if __name__ == "__main__":
    main()
