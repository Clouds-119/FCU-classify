# -*- coding: utf-8 -*-
"""
ensemble_window_models.py

用途：
在现有 window=120 类型化特征基础上，训练多个模型并做概率平均集成，
用于尽量降低单一 XGBoost 在低置信度样本上的风险。

适用目录：
artifacts_window_typed_v3_w120/
    window_train_features.csv
    window_test_features.csv
    window_schema.json

推荐先运行：
python ensemble_window_models.py ^
  --data_dir "artifacts_window_typed_v3_w120" ^
  --output_dir "artifacts_ensemble_w120" ^
  --k 300 ^
  --cv 5

输出：
artifacts_ensemble_w120/
    cv_model_summary.csv                      # 各模型CV结果
    oof_file_predictions_<model>.csv          # 各模型OOF文件级验证预测
    oof_low_confidence_summary.csv            # 低置信度验证分析
    test_file_predictions_<model>.csv         # 各模型测试集预测
    ensemble_uniform_file_predictions.csv     # 均匀平均集成结果
    ensemble_weighted_file_predictions.csv    # 按CV macro-F1加权集成结果
    ensemble_uniform_submission.csv           # 均匀集成提交
    ensemble_weighted_submission.csv          # 加权集成提交
    prediction_change_report.csv              # 单模型与集成结果差异
"""

from __future__ import annotations

import argparse
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. 数据读取
# ============================================================

def load_schema(data_dir: Path) -> dict:
    schema_file = data_dir / "window_schema.json"
    if not schema_file.exists():
        raise FileNotFoundError(f"找不到 schema 文件：{schema_file}")
    with open(schema_file, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_feature_columns(df: pd.DataFrame, schema: dict, is_train: bool = True) -> List[str]:
    """
    尽量从 schema 读取特征列；如果 schema 字段名不同，则自动排除元数据列。
    """
    candidate_keys = [
        "feature_names",
        "feature_columns",
        "selected_feature_names",
        "input_feature_names",
    ]

    for key in candidate_keys:
        if key in schema and isinstance(schema[key], list):
            cols = [c for c in schema[key] if c in df.columns]
            if len(cols) > 0:
                return cols

    meta_cols = {
        "label", "y", "group_id", "source_file", "window_id",
        "start_idx", "end_idx", "file_id", "class", "target"
    }
    cols = [c for c in df.columns if c not in meta_cols]
    return cols


def load_train_data(data_dir: Path):
    schema = load_schema(data_dir)
    train_file = data_dir / "window_train_features.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"找不到训练特征表：{train_file}")

    df = pd.read_csv(train_file)

    label_col = schema.get("label_col", "label")
    group_col = schema.get("group_col", "group_id")
    source_file_col = schema.get("source_file_col", "source_file")

    if label_col not in df.columns:
        if "label" in df.columns:
            label_col = "label"
        else:
            raise ValueError("训练特征表中找不到 label 列。")

    if group_col not in df.columns:
        if "group_id" in df.columns:
            group_col = "group_id"
        else:
            raise ValueError("训练特征表中找不到 group_id 列。")

    if source_file_col not in df.columns:
        source_file_col = group_col

    feature_cols = infer_feature_columns(df, schema, is_train=True)

    X = df[feature_cols].copy()
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    source_files = df[source_file_col].astype(str).to_numpy()

    return X, y, groups, source_files, feature_cols, schema


def load_test_data(data_dir: Path, feature_cols: List[str]):
    schema = load_schema(data_dir)
    test_file = data_dir / "window_test_features.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"找不到测试特征表：{test_file}")

    df = pd.read_csv(test_file)

    group_col = schema.get("group_col", "group_id")
    source_file_col = schema.get("source_file_col", "source_file")

    if group_col not in df.columns:
        if "group_id" in df.columns:
            group_col = "group_id"
        else:
            raise ValueError("测试特征表中找不到 group_id 列。")

    if source_file_col not in df.columns:
        source_file_col = group_col

    # 强制对齐训练特征列
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].copy()
    groups = df[group_col].astype(str).to_numpy()
    source_files = df[source_file_col].astype(str).to_numpy()

    meta = df[[group_col, source_file_col]].copy()
    if "window_id" in df.columns:
        meta["window_id"] = df["window_id"]
    else:
        meta["window_id"] = np.arange(len(df))

    meta = meta.rename(columns={group_col: "group_id", source_file_col: "source_file"})
    return X, groups, source_files, meta


# ============================================================
# 2. CV与聚合
# ============================================================

def get_group_labels(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"group": groups, "y": y})
    gids, glabels = [], []
    for g, sub in df.groupby("group", sort=False):
        vals = sub["y"].unique()
        if len(vals) != 1:
            raise ValueError(f"group={g} 内存在多个标签：{vals}")
        gids.append(g)
        glabels.append(vals[0])
    return np.asarray(gids), np.asarray(glabels)


def make_group_cv_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    _, group_labels = get_group_labels(y, groups)
    min_count = pd.Series(group_labels).value_counts().min()
    n_splits = min(n_splits, int(min_count))

    if n_splits < 2:
        raise ValueError("最小类别文件数不足2，无法进行分组交叉验证。")

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        return splits, "StratifiedGroupKFold", n_splits
    except Exception:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(np.zeros(len(y)), y, groups))
        return splits, "GroupKFold", n_splits


def align_proba(proba: np.ndarray, model_classes: List[int], global_classes: List[int]) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(global_classes)), dtype="float64")
    idx = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(global_classes):
        if c in idx:
            aligned[:, j] = proba[:, idx[c]]
    return aligned


def add_top_columns(df: pd.DataFrame, classes: List[int], prob_cols: List[str]) -> pd.DataFrame:
    P = df[prob_cols].to_numpy()
    order = np.argsort(P, axis=1)[:, ::-1]
    top1 = order[:, 0]

    df["predicted_label"] = [classes[i] for i in top1]
    df["top1_probability"] = P[np.arange(len(df)), top1]

    if len(classes) > 1:
        top2 = order[:, 1]
        df["top2_label"] = [classes[i] for i in top2]
        df["top2_probability"] = P[np.arange(len(df)), top2]
        df["prob_margin"] = df["top1_probability"] - df["top2_probability"]

    return df


def aggregate_to_file_level(
    proba: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    classes: List[int],
    y_window: np.ndarray | None = None,
) -> pd.DataFrame:
    prob_cols = [f"prob_class_{c}" for c in classes]

    df = pd.DataFrame(proba, columns=prob_cols)
    df["group_id"] = groups
    df["source_file"] = source_files

    if y_window is not None:
        df["label"] = y_window

    agg = {c: "mean" for c in prob_cols}
    agg["source_file"] = "first"
    if y_window is not None:
        agg["label"] = "first"

    out = df.groupby("group_id", sort=False).agg(agg).reset_index()
    out = add_top_columns(out, classes, prob_cols)
    return out


def file_sort_key(x: Any):
    stem = Path(str(x)).stem
    try:
        return (0, int(stem))
    except Exception:
        return (1, str(x))


# ============================================================
# 3. 模型定义
# ============================================================

def make_xgboost(random_state: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=random_state,
        n_jobs=-1,
    )


def make_catboost(random_state: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def make_lightgbm(random_state: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_random_forest(random_state: int):
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def make_extra_trees(random_state: int):
    return ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_pipeline(model, k: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(
            score_func=partial(mutual_info_classif, random_state=random_state),
            k=k,
        )),
        ("clf", model),
    ])


def build_model_dict(random_state: int, model_list: List[str]) -> Dict[str, Any]:
    makers = {
        "xgboost": make_xgboost,
        "catboost": make_catboost,
        "lightgbm": make_lightgbm,
        "randomforest": make_random_forest,
        "extratrees": make_extra_trees,
    }

    models = {}
    for name in model_list:
        key = name.lower()
        if key not in makers:
            print(f"跳过未知模型: {name}")
            continue
        try:
            models[key] = makers[key](random_state)
        except Exception as e:
            print(f"模型 {key} 初始化失败，已跳过：{e}")
    return models


# ============================================================
# 4. 评估、预测、集成
# ============================================================

def evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    classes: List[int],
    splits,
    k: int,
    random_state: int,
    output_dir: Path,
):
    file_pred_list = []
    window_metric_rows = []
    file_metric_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
        pipe = build_pipeline(model, k=k, random_state=random_state)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])

        proba = pipe.predict_proba(X.iloc[va_idx])
        proba = align_proba(proba, list(pipe.classes_), classes)

        # 窗口级验证：更接近测试集单窗口形态
        win_pred = np.asarray([classes[i] for i in np.argmax(proba, axis=1)])
        window_metric_rows.append({
            "model": model_name,
            "fold": fold_id,
            "level": "window",
            "n_samples": int(len(va_idx)),
            "accuracy": float(accuracy_score(y[va_idx], win_pred)),
            "macro_f1": float(f1_score(y[va_idx], win_pred, average="macro")),
            "weighted_f1": float(f1_score(y[va_idx], win_pred, average="weighted")),
        })

        # 文件级验证：和原先一致，多个窗口概率平均
        file_pred = aggregate_to_file_level(
            proba=proba,
            groups=groups[va_idx],
            source_files=source_files[va_idx],
            classes=classes,
            y_window=y[va_idx],
        )
        file_pred["fold"] = fold_id
        file_pred["model"] = model_name

        y_true_file = file_pred["label"].to_numpy()
        y_pred_file = file_pred["predicted_label"].to_numpy()

        file_metric_rows.append({
            "model": model_name,
            "fold": fold_id,
            "level": "file",
            "n_samples": int(len(file_pred)),
            "accuracy": float(accuracy_score(y_true_file, y_pred_file)),
            "macro_f1": float(f1_score(y_true_file, y_pred_file, average="macro")),
            "weighted_f1": float(f1_score(y_true_file, y_pred_file, average="weighted")),
        })

        file_pred_list.append(file_pred)

    file_pred_all = pd.concat(file_pred_list, ignore_index=True)

    file_pred_path = output_dir / f"oof_file_predictions_{model_name}.csv"
    file_pred_all.to_csv(file_pred_path, index=False, encoding="utf-8-sig")

    all_metrics = pd.DataFrame(window_metric_rows + file_metric_rows)
    summary = []
    for level in ["window", "file"]:
        sub = all_metrics[all_metrics["level"] == level]
        summary.append({
            "model": model_name,
            "level": level,
            "accuracy_mean": sub["accuracy"].mean(),
            "accuracy_std": sub["accuracy"].std(ddof=0),
            "macro_f1_mean": sub["macro_f1"].mean(),
            "macro_f1_std": sub["macro_f1"].std(ddof=0),
            "weighted_f1_mean": sub["weighted_f1"].mean(),
            "weighted_f1_std": sub["weighted_f1"].std(ddof=0),
            "oof_file_predictions": str(file_pred_path),
        })

    return pd.DataFrame(summary), all_metrics, file_pred_all


def train_predict_test(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    test_groups: np.ndarray,
    test_source_files: np.ndarray,
    classes: List[int],
    k: int,
    random_state: int,
    output_dir: Path,
):
    pipe = build_pipeline(model, k=k, random_state=random_state)
    pipe.fit(X_train, y)

    proba = pipe.predict_proba(X_test)
    proba = align_proba(proba, list(pipe.classes_), classes)

    file_pred = aggregate_to_file_level(
        proba=proba,
        groups=test_groups,
        source_files=test_source_files,
        classes=classes,
        y_window=None,
    )

    file_pred = file_pred.sort_values(
        by="source_file",
        key=lambda s: s.map(file_sort_key),
    ).reset_index(drop=True)

    out_path = output_dir / f"test_file_predictions_{model_name}.csv"
    file_pred.to_csv(out_path, index=False, encoding="utf-8-sig")

    model_path = output_dir / f"final_model_{model_name}.pkl"
    joblib.dump({
        "pipeline": pipe,
        "model_name": model_name,
        "k": int(k),
        "classes": classes,
    }, model_path)

    return file_pred, out_path, model_path


def ensemble_predictions(
    pred_dict: Dict[str, pd.DataFrame],
    classes: List[int],
    output_dir: Path,
    name: str,
    weights: Dict[str, float] | None = None,
):
    prob_cols = [f"prob_class_{c}" for c in classes]
    model_names = list(pred_dict.keys())

    base = pred_dict[model_names[0]][["group_id", "source_file"]].copy()
    P_sum = np.zeros((len(base), len(classes)), dtype="float64")

    if weights is None:
        weights = {m: 1.0 / len(model_names) for m in model_names}
    else:
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 0:
            weights = {m: 1.0 / len(model_names) for m in model_names}
        else:
            weights = {m: max(0.0, float(v)) / total for m, v in weights.items()}

    for m in model_names:
        df = pred_dict[m].sort_values(
            by="source_file",
            key=lambda s: s.map(file_sort_key),
        ).reset_index(drop=True)

        if not (df["source_file"].astype(str).tolist() == base["source_file"].astype(str).tolist()):
            raise ValueError(f"{m} 的测试文件顺序与基准模型不一致。")

        P_sum += weights.get(m, 0.0) * df[prob_cols].to_numpy()

    ens = base.copy()
    for j, c in enumerate(classes):
        ens[f"prob_class_{c}"] = P_sum[:, j]

    ens = add_top_columns(ens, classes, prob_cols)

    file_path = output_dir / f"{name}_file_predictions.csv"
    sub_path = output_dir / f"{name}_submission.csv"

    ens.to_csv(file_path, index=False, encoding="utf-8-sig")
    ens[["source_file", "predicted_label"]].rename(
        columns={"source_file": "filename", "predicted_label": "label"}
    ).to_csv(sub_path, index=False, encoding="utf-8-sig")

    return ens, file_path, sub_path, weights


def low_confidence_summary(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    rows = []
    if label_col in df.columns:
        df = df.copy()
        df["correct"] = df[label_col] == df["predicted_label"]

        rows.append({
            "condition": "all",
            "n": len(df),
            "accuracy": df["correct"].mean(),
        })

        for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            sub = df[df["top1_probability"] < th]
            rows.append({
                "condition": f"top1_probability < {th}",
                "n": len(sub),
                "accuracy": sub["correct"].mean() if len(sub) else np.nan,
            })

        for th in [0.05, 0.1, 0.2, 0.3]:
            sub = df[df["prob_margin"] < th]
            rows.append({
                "condition": f"prob_margin < {th}",
                "n": len(sub),
                "accuracy": sub["correct"].mean() if len(sub) else np.nan,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="多模型概率集成，提高测试集稳健性")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts_ensemble_w120")
    parser.add_argument("--k", type=int, default=300)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="xgboost,catboost,lightgbm,randomforest,extratrees",
        help="候选模型，用逗号分隔。可选：xgboost,catboost,lightgbm,randomforest,extratrees",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, source_files, feature_cols, schema = load_train_data(data_dir)
    X_test, test_groups, test_source_files, test_meta = load_test_data(data_dir, feature_cols)

    classes = sorted(np.unique(y).tolist())

    print("\n========== 数据检查 ==========")
    print(f"训练窗口样本数: {X.shape[0]}")
    print(f"训练特征维度: {X.shape[1]}")
    print(f"训练原始文件数: {len(np.unique(groups))}")
    print(f"测试窗口样本数: {X_test.shape[0]}")
    print(f"测试文件数: {len(np.unique(test_groups))}")
    print(f"类别列表: {classes}")
    print("文件级类别分布:", dict(pd.Series(get_group_labels(y, groups)[1]).value_counts().sort_index()))
    print("窗口级类别分布:", dict(pd.Series(y).value_counts().sort_index()))

    splits, cv_name, n_splits = make_group_cv_splits(
        y=y,
        groups=groups,
        n_splits=args.cv,
        random_state=args.random_state,
    )

    print("\n========== CV 设置 ==========")
    print(f"CV 方法: {cv_name}")
    print(f"折数: {n_splits}")
    print("说明：同时输出窗口级验证和文件级聚合验证。")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    models = build_model_dict(args.random_state, model_list)
    if not models:
        raise RuntimeError("没有可用模型，请检查依赖是否安装。")

    all_summary = []
    all_fold_metrics = []
    all_low_conf = []
    test_pred_dict = {}
    cv_weight_source = {}

    print("\n========== 开始多模型训练与预测 ==========")
    for model_name, model in models.items():
        print(f"\n----- 模型：{model_name} -----")

        summary_df, fold_metric_df, oof_file_pred = evaluate_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            groups=groups,
            source_files=source_files,
            classes=classes,
            splits=splits,
            k=args.k,
            random_state=args.random_state,
            output_dir=output_dir,
        )

        print(summary_df.to_string(index=False))
        all_summary.append(summary_df)
        all_fold_metrics.append(fold_metric_df)

        low_df = low_confidence_summary(oof_file_pred, label_col="label")
        low_df.insert(0, "model", model_name)
        all_low_conf.append(low_df)

        # 用文件级 macro-F1 作为加权集成权重来源
        file_row = summary_df[summary_df["level"] == "file"].iloc[0]
        cv_weight_source[model_name] = float(file_row["macro_f1_mean"])

        test_pred, pred_path, model_path = train_predict_test(
            model_name=model_name,
            model=model,
            X_train=X,
            y=y,
            X_test=X_test,
            test_groups=test_groups,
            test_source_files=test_source_files,
            classes=classes,
            k=args.k,
            random_state=args.random_state,
            output_dir=output_dir,
        )

        test_pred_dict[model_name] = test_pred
        print(f"测试预测已保存: {pred_path}")
        print(f"最终模型已保存: {model_path}")

    summary_all = pd.concat(all_summary, ignore_index=True)
    fold_metrics_all = pd.concat(all_fold_metrics, ignore_index=True)
    low_conf_all = pd.concat(all_low_conf, ignore_index=True)

    summary_path = output_dir / "cv_model_summary.csv"
    fold_metric_path = output_dir / "cv_fold_metrics.csv"
    low_conf_path = output_dir / "oof_low_confidence_summary.csv"

    summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fold_metrics_all.to_csv(fold_metric_path, index=False, encoding="utf-8-sig")
    low_conf_all.to_csv(low_conf_path, index=False, encoding="utf-8-sig")

    print("\n========== 多模型 CV 汇总 ==========")
    print(summary_all.sort_values(["level", "macro_f1_mean"], ascending=[True, False]).to_string(index=False))

    # 集成1：均匀平均
    ens_uniform, ens_uniform_path, ens_uniform_sub, uniform_weights = ensemble_predictions(
        pred_dict=test_pred_dict,
        classes=classes,
        output_dir=output_dir,
        name="ensemble_uniform",
        weights=None,
    )

    # 集成2：按CV文件级macro-F1加权
    ens_weighted, ens_weighted_path, ens_weighted_sub, norm_weights = ensemble_predictions(
        pred_dict=test_pred_dict,
        classes=classes,
        output_dir=output_dir,
        name="ensemble_weighted",
        weights=cv_weight_source,
    )

    # 变化报告：和 xgboost 单模型对比
    change_rows = []
    if "xgboost" in test_pred_dict:
        xgb = test_pred_dict["xgboost"].sort_values(
            by="source_file",
            key=lambda s: s.map(file_sort_key),
        ).reset_index(drop=True)

        for ens_name, ens_df in [("ensemble_uniform", ens_uniform), ("ensemble_weighted", ens_weighted)]:
            tmp = pd.DataFrame({
                "source_file": xgb["source_file"],
                "xgboost_label": xgb["predicted_label"],
                f"{ens_name}_label": ens_df["predicted_label"],
                "xgboost_top1_probability": xgb["top1_probability"],
                "xgboost_prob_margin": xgb["prob_margin"],
                f"{ens_name}_top1_probability": ens_df["top1_probability"],
                f"{ens_name}_prob_margin": ens_df["prob_margin"],
            })
            tmp["changed"] = tmp["xgboost_label"] != tmp[f"{ens_name}_label"]
            tmp["ensemble_name"] = ens_name
            change_rows.append(tmp)

    if change_rows:
        change_report = pd.concat(change_rows, ignore_index=True)
        change_path = output_dir / "prediction_change_report.csv"
        change_report.to_csv(change_path, index=False, encoding="utf-8-sig")
    else:
        change_path = None

    # 预测分布与低置信度
    def pred_info(df, name):
        return {
            "name": name,
            "n_files": int(len(df)),
            "distribution": {str(k): int(v) for k, v in df["predicted_label"].value_counts().sort_index().to_dict().items()},
            "top1_probability_lt_0.6": int((df["top1_probability"] < 0.6).sum()),
            "prob_margin_lt_0.1": int((df["prob_margin"] < 0.1).sum()),
            "top1_probability_mean": float(df["top1_probability"].mean()),
            "prob_margin_mean": float(df["prob_margin"].mean()),
        }

    info = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "k": int(args.k),
        "cv_method": cv_name,
        "n_splits": int(n_splits),
        "models": list(models.keys()),
        "cv_weight_source": cv_weight_source,
        "normalized_weighted_ensemble_weights": norm_weights,
        "outputs": {
            "cv_model_summary": str(summary_path),
            "cv_fold_metrics": str(fold_metric_path),
            "oof_low_confidence_summary": str(low_conf_path),
            "ensemble_uniform_file_predictions": str(ens_uniform_path),
            "ensemble_uniform_submission": str(ens_uniform_sub),
            "ensemble_weighted_file_predictions": str(ens_weighted_path),
            "ensemble_weighted_submission": str(ens_weighted_sub),
            "prediction_change_report": str(change_path) if change_path else None,
        },
        "test_prediction_info": [
            pred_info(df, name) for name, df in test_pred_dict.items()
        ] + [
            pred_info(ens_uniform, "ensemble_uniform"),
            pred_info(ens_weighted, "ensemble_weighted"),
        ],
    }

    info_path = output_dir / "ensemble_run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 集成完成 ==========")
    print(f"均匀集成预测: {ens_uniform_path}")
    print(f"均匀集成提交: {ens_uniform_sub}")
    print(f"加权集成预测: {ens_weighted_path}")
    print(f"加权集成提交: {ens_weighted_sub}")
    print(f"CV 汇总: {summary_path}")
    print(f"低置信度分析: {low_conf_path}")
    if change_path:
        print(f"与 XGBoost 变化报告: {change_path}")
    print(f"运行信息: {info_path}")

    print("\n推荐查看：")
    print("1. cv_model_summary.csv：看哪些模型单窗口/文件级验证更稳。")
    print("2. prediction_change_report.csv：看集成相对 XGBoost 改了哪些样本。")
    print("3. ensemble_uniform_submission.csv 与 ensemble_weighted_submission.csv：两个备选提交。")


if __name__ == "__main__":
    main()
