# -*- coding: utf-8 -*-
"""
window_train_only.py

窗口级 FCU 故障分类训练脚本。

输入：
    artifacts_window_data/window_train_features.csv
    artifacts_window_data/window_schema.json

输出：
    artifacts_window_train/window_cv_results.csv
    artifacts_window_train/window_cv_file_predictions.csv
    artifacts_window_train/best_window_pipeline.pkl
    artifacts_window_train/window_train_info.json

核心原则：
    训练样本是“窗口”，但验证评价必须回到“文件级”。
    交叉验证必须使用 group_id=原始文件名 分组，避免同一文件的窗口同时进入训练集和验证集。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. 基础工具
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


def load_train_data(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """
    读取窗口级训练特征表。
    """
    schema = load_schema(data_dir)
    train_file = data_dir / "window_train_features.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"找不到窗口训练特征表：{train_file}")

    df = pd.read_csv(train_file)

    label_col = schema.get("label_col", "label")
    group_col = schema.get("group_col", "group_id")
    feature_names = schema["feature_names"]

    for c in [label_col, group_col, "source_file"]:
        if c not in df.columns:
            raise KeyError(f"窗口训练特征表缺少必要列：{c}")

    X = df.reindex(columns=feature_names)
    y = df[label_col].to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    source_files = df["source_file"].astype(str).to_numpy()

    return X, y, groups, source_files, feature_names, schema


def get_group_labels(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取文件级 group 标签。
    同一个 group 内所有窗口标签应一致。
    """
    df = pd.DataFrame({"group": groups, "y": y})
    group_ids = []
    group_labels = []

    for g, sub in df.groupby("group", sort=False):
        labels = sub["y"].unique()
        if len(labels) != 1:
            raise ValueError(f"group={g} 内出现多个标签：{labels}")
        group_ids.append(g)
        group_labels.append(labels[0])

    return np.asarray(group_ids), np.asarray(group_labels)


def make_group_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
):
    """
    优先使用 StratifiedGroupKFold。
    如果 sklearn 版本不支持，则退回 GroupKFold。
    """
    _, group_labels = get_group_labels(y, groups)
    min_count = pd.Series(group_labels).value_counts().min()
    n_splits = min(n_splits, int(min_count))

    if n_splits < 2:
        raise ValueError("最小类别文件数不足 2，无法进行分组交叉验证。")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        cv_name = "StratifiedGroupKFold"

    except Exception:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        cv_name = "GroupKFold"

    return splits, cv_name, n_splits


# ============================================================
# 2. 文件级概率聚合
# ============================================================

def align_proba(proba: np.ndarray, model_classes: List[int], global_classes: List[int]) -> np.ndarray:
    """
    将不同模型输出概率列对齐到统一类别顺序。
    """
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype="float64")
    class_to_idx = {c: i for i, c in enumerate(model_classes)}

    for j, c in enumerate(global_classes):
        if c in class_to_idx:
            aligned[:, j] = proba[:, class_to_idx[c]]

    return aligned


def aggregate_to_file_level(
    proba: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    classes: List[int],
    y_window: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    将窗口级概率按原始文件 group_id 平均，得到文件级概率和预测类别。
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
# 3. 模型与 Pipeline
# ============================================================

def build_models(random_state: int = 42) -> Dict[str, object]:
    """
    多模型候选池。
    """
    models: Dict[str, object] = {}

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
    窗口级训练 Pipeline：
    缺失值填补 -> 零方差删除 -> 互信息特征筛选 -> 分类模型
    """
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
        k=k,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", selector),
        ("clf", model),
    ])


# ============================================================
# 4. 交叉验证评估
# ============================================================

def evaluate_model(
    model_name: str,
    model,
    k: int,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    splits,
    global_classes: List[int],
    random_state: int,
    save_fold_pred: bool,
    output_dir: Path,
) -> Tuple[dict, pd.DataFrame]:
    """
    在分组交叉验证下评估一个模型。
    训练是窗口级，评价是文件级。
    """
    fold_metrics = []
    fold_file_preds = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
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
            classes = global_classes
            proba = np.zeros((len(pred), len(classes)), dtype="float64")
            c_to_i = {c: i for i, c in enumerate(classes)}
            for i, p in enumerate(pred):
                proba[i, c_to_i[p]] = 1.0

        proba = align_proba(proba, classes, global_classes)

        file_pred = aggregate_to_file_level(
            proba=proba,
            groups=groups_valid,
            source_files=source_valid,
            classes=global_classes,
            y_window=y_valid,
        )

        y_true_file = file_pred["label"].to_numpy()
        y_pred_file = file_pred["predicted_label"].to_numpy()

        acc = accuracy_score(y_true_file, y_pred_file)
        macro_f1 = f1_score(y_true_file, y_pred_file, average="macro")
        weighted_f1 = f1_score(y_true_file, y_pred_file, average="weighted")

        fold_metrics.append({
            "fold": fold_id,
            "n_valid_files": int(len(file_pred)),
            "n_valid_windows": int(len(valid_idx)),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        })

        file_pred["fold"] = fold_id
        file_pred["model"] = model_name
        file_pred["k"] = k
        fold_file_preds.append(file_pred)

    fold_df = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(fold_file_preds, ignore_index=True)

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

    if save_fold_pred:
        pred_path = output_dir / f"cv_file_pred_{model_name}_k{k}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return result, pred_df


# ============================================================
# 5. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="窗口级 FCU 多模型训练脚本")
    parser.add_argument("--data_dir", type=str, default="artifacts_window_data")
    parser.add_argument("--output_dir", type=str, default="artifacts_window_train")
    parser.add_argument("--k_list", type=str, default="300,400,500")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_fold_predictions", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, source_files, feature_names, schema = load_train_data(data_dir)
    global_classes = sorted(np.unique(y).tolist())

    print("\n========== 窗口训练数据检查 ==========")
    print(f"窗口样本数: {X.shape[0]}")
    print(f"窗口特征维度: {X.shape[1]}")
    print(f"原始文件数: {len(np.unique(groups))}")
    print(f"类别列表: {global_classes}")
    print("窗口级类别分布:", dict(pd.Series(y).value_counts().sort_index()))
    print("文件级类别分布:", dict(pd.Series(get_group_labels(y, groups)[1]).value_counts().sort_index()))
    print(f"缺失值比例均值: {float(X.isna().mean().mean()):.6f}")

    splits, cv_name, n_splits = make_group_cv_splits(
        y=y,
        groups=groups,
        n_splits=args.cv,
        random_state=args.random_state,
    )

    print("\n========== 交叉验证设置 ==========")
    print(f"CV 方法: {cv_name}")
    print(f"折数: {n_splits}")
    print("注意：同一原始文件切出的所有窗口不会同时进入训练集和验证集。")

    models = build_models(random_state=args.random_state)
    k_list = parse_int_list(args.k_list)

    results = []
    all_oof_preds = []

    best_score = -np.inf
    best_model_name = None
    best_k = None
    best_model = None

    print("\n========== 开始窗口级多模型训练，文件级验证 ==========")

    for k in k_list:
        for model_name, model in models.items():
            exp_name = f"{model_name}_k{k}"
            print(f"\n正在评估：{exp_name}")

            try:
                result, pred_df = evaluate_model(
                    model_name=model_name,
                    model=model,
                    k=k,
                    X=X,
                    y=y,
                    groups=groups,
                    source_files=source_files,
                    splits=splits,
                    global_classes=global_classes,
                    random_state=args.random_state,
                    save_fold_pred=args.save_fold_predictions,
                    output_dir=output_dir,
                )

                print(
                    f"  accuracy={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}, "
                    f"macro-F1={result['macro_f1_mean']:.4f}±{result['macro_f1_std']:.4f}, "
                    f"weighted-F1={result['weighted_f1_mean']:.4f}±{result['weighted_f1_std']:.4f}"
                )

                results.append(result)

                if result["macro_f1_mean"] > best_score:
                    best_score = result["macro_f1_mean"]
                    best_model_name = model_name
                    best_k = k
                    best_model = clone(model)
                    best_pred_df = pred_df.copy()
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
        raise RuntimeError("所有模型均训练失败，请检查数据和依赖。")

    # 保存最优 OOF 文件级预测
    oof_path = output_dir / "best_cv_file_predictions.csv"
    best_pred_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    # 输出最优模型文件级分类报告与混淆矩阵
    y_true_best = best_pred_df["label"].to_numpy()
    y_pred_best = best_pred_df["predicted_label"].to_numpy()

    report = classification_report(
        y_true_best,
        y_pred_best,
        labels=global_classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_path = output_dir / "best_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm = confusion_matrix(y_true_best, y_pred_best, labels=global_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in global_classes],
        columns=[f"pred_{c}" for c in global_classes],
    )
    cm_path = output_dir / "best_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print("\n========== 使用全部窗口样本训练最终模型 ==========")
    print(f"最优模型: {best_model_name}")
    print(f"最优 k: {best_k}")
    print(f"最优文件级 macro-F1: {best_score:.4f}")

    best_pipe = build_pipeline(best_model, k=best_k, random_state=args.random_state)
    best_pipe.fit(X, y)

    model_path = output_dir / "best_window_pipeline.pkl"

    artifact = {
        "pipeline": best_pipe,
        "model_name": best_model_name,
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "feature_names": feature_names,
        "class_labels": global_classes,
        "schema": schema,
        "cv_results": results,
        "aggregation": "mean_probability_by_group",
        "group_col": schema.get("group_col", "group_id"),
        "source_file_col": "source_file",
    }

    joblib.dump(artifact, model_path)

    info = {
        "model_name": best_model_name,
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "n_window_samples": int(X.shape[0]),
        "n_files": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "cv_method": cv_name,
        "n_splits": int(n_splits),
        "class_labels": global_classes,
        "outputs": {
            "model": str(model_path),
            "cv_results": str(cv_path),
            "oof_file_predictions": str(oof_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(cm_path),
        },
    }

    info_path = output_dir / "window_train_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 训练完成 ==========")
    print(f"模型文件: {model_path}")
    print(f"交叉验证结果: {cv_path}")
    print(f"OOF 文件级预测: {oof_path}")
    print(f"分类报告: {report_path}")
    print(f"混淆矩阵: {cm_path}")
    print(f"训练信息: {info_path}")


if __name__ == "__main__":
    main()
