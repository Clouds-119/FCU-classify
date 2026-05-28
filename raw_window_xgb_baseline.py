# -*- coding: utf-8 -*-
"""
raw_window_xgb_baseline.py

Raw-Window Flatten + XGBoost 基线脚本。

用途：
构建一版“几乎不做人工特征工程”的基线模型：
1. 对训练集每个 2 天文件切分 120min 窗口；
2. 不计算均值、方差、状态占比、风机一致性等人工统计特征；
3. 直接将窗口内原始多变量序列按时间顺序展开为固定长度向量；
4. 使用 SelectKBest + XGBoost 做分类；
5. 仍然使用 StratifiedGroupKFold，并按原始文件做文件级验证；
6. 预测测试集时，每个 120 行测试文件直接作为一个窗口输入模型。

推荐训练：
python raw_window_xgb_baseline.py train ^
  --train_root ".\\大作业2数据\\训练集" ^
  --output_dir "artifacts_raw_window_xgb" ^
  --window_len 120 ^
  --stride 30 ^
  --k_list 200,300,500,800 ^
  --cv 5

推荐预测：
python raw_window_xgb_baseline.py predict ^
  --test_root ".\\大作业2数据\\测试集" ^
  --model_file "artifacts_raw_window_xgb\\best_raw_window_xgb.pkl" ^
  --output_dir "test_results_raw_window"

输出：
训练阶段：
- cv_results_raw_window_xgb.csv
- best_raw_window_xgb.pkl
- best_cv_file_predictions.csv
- raw_window_schema.json

预测阶段：
- raw_window_file_predictions.csv
- raw_window_submission.csv
- raw_window_level_predictions.csv
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
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))


def find_data_files(root: Path, recursive: bool = True) -> List[Path]:
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat) if recursive else root.glob(pat))
    return sorted(files)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        # 优先 utf-8-sig；失败再尝试 gbk
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    return pd.read_excel(path)


def guess_time_columns(columns: List[str]) -> List[str]:
    time_keywords = [
        "time", "timestamp", "date", "datetime",
        "时间", "日期", "时刻"
    ]
    out = []
    for c in columns:
        cl = str(c).lower()
        if any(k in cl for k in time_keywords):
            out.append(c)
    return out


def infer_feature_columns(train_files: List[Path]) -> Tuple[List[str], List[str]]:
    """
    从第一个训练文件推断特征列。
    默认删除时间戳列，其余列都作为原始输入变量。
    """
    if not train_files:
        raise FileNotFoundError("没有找到训练文件。")

    df0 = read_table(train_files[0])
    cols = list(df0.columns)
    time_cols = guess_time_columns(cols)
    feature_cols = [c for c in cols if c not in time_cols]

    if len(feature_cols) == 0:
        raise ValueError("没有推断出有效特征列，请检查 CSV 表头。")

    return feature_cols, time_cols


def fit_encoders(train_files: List[Path], feature_cols: List[str], max_files: int | None = None) -> Dict[str, Dict[str, int]]:
    """
    对非数值列建立类别编码映射。
    如果原始列本身都是数字，这里返回空映射。
    """
    encoders: Dict[str, Dict[str, int]] = {c: {} for c in feature_cols}

    files = train_files if max_files is None else train_files[:max_files]
    for p in files:
        df = read_table(p)
        for c in feature_cols:
            if c not in df.columns:
                continue
            s = df[c]
            # 如果能整体转成数值，则不作为类别列
            numeric = pd.to_numeric(s, errors="coerce")
            non_na_ratio = numeric.notna().mean()
            if non_na_ratio > 0.95:
                continue

            vals = s.dropna().astype(str).unique().tolist()
            mp = encoders[c]
            for v in vals:
                if v not in mp:
                    mp[v] = len(mp)

    # 去掉空映射
    encoders = {c: mp for c, mp in encoders.items() if len(mp) > 0}
    return encoders


def dataframe_to_numeric_array(
    df: pd.DataFrame,
    feature_cols: List[str],
    encoders: Dict[str, Dict[str, int]],
) -> np.ndarray:
    """
    将原始 DataFrame 转成数值矩阵。
    - 数值列：to_numeric
    - 类别列：用训练集映射编码，未知类别为 -1
    """
    arrays = []
    for c in feature_cols:
        if c not in df.columns:
            arr = np.full(len(df), np.nan, dtype="float32")
        elif c in encoders:
            mp = encoders[c]
            arr = df[c].astype(str).map(mp).fillna(-1).astype("float32").to_numpy()
        else:
            arr = pd.to_numeric(df[c], errors="coerce").astype("float32").to_numpy()
        arrays.append(arr)

    if not arrays:
        raise ValueError("没有可用特征列。")

    X = np.vstack(arrays).T.astype("float32")
    return X


def make_window_slices(n_rows: int, window_len: int, stride: int, allow_short_last: bool = False) -> List[Tuple[int, int]]:
    if n_rows >= window_len:
        return [(start, start + window_len) for start in range(0, n_rows - window_len + 1, stride)]

    if allow_short_last and n_rows >= int(window_len * 0.8):
        return [(0, n_rows)]

    return []


def flatten_window(arr: np.ndarray, window_len: int) -> np.ndarray:
    """
    将 [T, C] 展平为 [T*C]。
    若 T < window_len，则尾部 padding NaN。
    """
    T, C = arr.shape
    if T < window_len:
        pad = np.full((window_len - T, C), np.nan, dtype="float32")
        arr = np.vstack([arr, pad])
    elif T > window_len:
        arr = arr[:window_len]
    return arr.reshape(-1).astype("float32")


def make_flatten_feature_names(feature_cols: List[str], window_len: int) -> List[str]:
    names = []
    for t in range(window_len):
        for c in feature_cols:
            names.append(f"t{t:03d}__{c}")
    return names


# ============================================================
# 2. 构造训练/测试 Raw Window Flatten 数据
# ============================================================

def build_train_raw_window_dataset(
    train_root: Path,
    window_len: int,
    stride: int,
    allow_short_last: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], dict]:
    """
    训练集结构要求：
    train_root/
        0/*.csv
        1/*.csv
        ...
        8/*.csv
    """
    train_files = []
    labels = []

    for class_dir in sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda x: x.name):
        try:
            label = int(class_dir.name)
        except ValueError:
            continue

        files = find_data_files(class_dir, recursive=False)
        print(f"类别 {label} 文件数量: {len(files)}")
        for f in files:
            train_files.append(f)
            labels.append(label)

    if not train_files:
        raise FileNotFoundError(f"没有在训练目录中找到类别子文件夹和数据文件：{train_root}")

    feature_cols, time_cols = infer_feature_columns(train_files)
    encoders = fit_encoders(train_files, feature_cols)

    print("\n========== Raw-Window Flatten 数据设置 ==========")
    print(f"删除时间列: {time_cols if time_cols else '无'}")
    print(f"原始输入变量数: {len(feature_cols)}")
    print(f"类别编码列数: {len(encoders)}")
    if encoders:
        print("类别编码列示例:", list(encoders.keys())[:10])

    X_rows = []
    y_rows = []
    group_rows = []
    source_rows = []
    win_id_rows = []

    for p, label in zip(train_files, labels):
        df = read_table(p)
        arr = dataframe_to_numeric_array(df, feature_cols, encoders)
        slices = make_window_slices(len(df), window_len, stride, allow_short_last=allow_short_last)

        for wi, (s, e) in enumerate(slices):
            flat = flatten_window(arr[s:e], window_len)
            X_rows.append(flat)
            y_rows.append(label)
            group_rows.append(str(p.relative_to(train_root)))
            source_rows.append(p.name)
            win_id_rows.append(wi)

    if not X_rows:
        raise RuntimeError("没有生成任何训练窗口，请检查 window_len 和 stride。")

    X_np = np.vstack(X_rows).astype("float32")
    feature_names = make_flatten_feature_names(feature_cols, window_len)

    X_df = pd.DataFrame(X_np, columns=feature_names)
    y = np.asarray(y_rows, dtype=int)
    groups = np.asarray(group_rows, dtype=str)
    source_files = np.asarray(source_rows, dtype=str)

    meta_df = pd.DataFrame({
        "group_id": groups,
        "source_file": source_files,
        "window_id": win_id_rows,
        "label": y,
    })

    schema = {
        "mode": "raw_window_flatten",
        "window_len": int(window_len),
        "stride": int(stride),
        "feature_cols": [str(c) for c in feature_cols],
        "time_cols": [str(c) for c in time_cols],
        "encoders": encoders,
        "feature_names": feature_names,
        "label_col": "label",
        "group_col": "group_id",
        "source_file_col": "source_file",
    }

    return X_df, y, groups, source_files, feature_names, schema, meta_df


def build_test_raw_window_dataset(
    test_root: Path,
    schema: dict,
    allow_short_last: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    window_len = int(schema["window_len"])
    stride = int(schema["stride"])
    feature_cols = schema["feature_cols"]
    encoders = schema.get("encoders", {})
    feature_names = schema["feature_names"]

    test_files = find_data_files(test_root, recursive=False)
    if not test_files:
        raise FileNotFoundError(f"测试目录下没有找到 csv/xlsx 文件：{test_root}")

    X_rows = []
    group_rows = []
    source_rows = []
    win_id_rows = []

    for p in test_files:
        df = read_table(p)
        arr = dataframe_to_numeric_array(df, feature_cols, encoders)
        slices = make_window_slices(len(df), window_len, stride, allow_short_last=allow_short_last)

        # 测试集如果正好 120 行，应该生成 1 个窗口
        if not slices and len(df) > 0:
            if len(df) == window_len:
                slices = [(0, window_len)]

        for wi, (s, e) in enumerate(slices):
            flat = flatten_window(arr[s:e], window_len)
            X_rows.append(flat)
            group_rows.append(p.name)
            source_rows.append(p.name)
            win_id_rows.append(wi)

    if not X_rows:
        raise RuntimeError("没有生成任何测试窗口，请检查测试文件长度与 window_len。")

    X_np = np.vstack(X_rows).astype("float32")
    X_df = pd.DataFrame(X_np, columns=feature_names)

    groups = np.asarray(group_rows, dtype=str)
    source_files = np.asarray(source_rows, dtype=str)

    meta_df = pd.DataFrame({
        "group_id": groups,
        "source_file": source_files,
        "window_id": win_id_rows,
    })

    return X_df, groups, source_files, feature_names, meta_df


# ============================================================
# 3. CV、聚合、模型
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
        raise ValueError("最小类别文件数不足 2，无法进行分组交叉验证。")

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
    mp = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(global_classes):
        if c in mp:
            aligned[:, j] = proba[:, mp[c]]
    return aligned


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
    P = out[prob_cols].to_numpy()
    order = np.argsort(P, axis=1)[:, ::-1]

    top1 = order[:, 0]
    out["predicted_label"] = [classes[i] for i in top1]
    out["top1_probability"] = P[np.arange(len(out)), top1]

    if len(classes) > 1:
        top2 = order[:, 1]
        out["top2_label"] = [classes[i] for i in top2]
        out["top2_probability"] = P[np.arange(len(out)), top2]
        out["prob_margin"] = out["top1_probability"] - out["top2_probability"]

    return out


def make_xgb(random_state: int = 42):
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


def build_pipeline(k: int, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(
            score_func=partial(mutual_info_classif, random_state=random_state),
            k=k,
        )),
        ("clf", make_xgb(random_state=random_state)),
    ])


def evaluate_setting(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    source_files: np.ndarray,
    k: int,
    splits,
    global_classes: List[int],
    random_state: int,
):
    fold_rows = []
    pred_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
        pipe = build_pipeline(k=k, random_state=random_state)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])

        proba = pipe.predict_proba(X.iloc[va_idx])
        proba = align_proba(proba, list(pipe.classes_), global_classes)

        file_pred = aggregate_to_file_level(
            proba=proba,
            groups=groups[va_idx],
            source_files=source_files[va_idx],
            classes=global_classes,
            y_window=y[va_idx],
        )

        y_true = file_pred["label"].to_numpy()
        y_pred = file_pred["predicted_label"].to_numpy()

        fold_rows.append({
            "fold": fold_id,
            "n_valid_files": int(len(file_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        })

        file_pred["fold"] = fold_id
        file_pred["k"] = k
        pred_rows.append(file_pred)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    res = {
        "k": int(k),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(fold_df["macro_f1"].mean()),
        "macro_f1_std": float(fold_df["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
        "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=0)),
    }
    return res, pred_df


# ============================================================
# 4. train / predict
# ============================================================

def run_train(args):
    train_root = Path(args.train_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== 读取训练集并构造 Raw-Window Flatten 样本 ==========")
    X, y, groups, source_files, feature_names, schema, meta_df = build_train_raw_window_dataset(
        train_root=train_root,
        window_len=args.window_len,
        stride=args.stride,
        allow_short_last=args.allow_short_last,
    )

    print("\n========== 数据检查 ==========")
    print(f"窗口样本数: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    print(f"原始文件数: {len(np.unique(groups))}")
    print("文件级类别分布:", dict(pd.Series(get_group_labels(y, groups)[1]).value_counts().sort_index()))
    print("窗口级类别分布:", dict(pd.Series(y).value_counts().sort_index()))
    print(f"缺失值比例均值: {float(X.isna().mean().mean()):.6f}")

    # 保存构造后的特征与 schema，便于排查
    train_feature_path = output_dir / "raw_window_train_features.csv"
    train_meta_path = output_dir / "raw_window_train_meta.csv"
    schema_path = output_dir / "raw_window_schema.json"

    # 为避免 CSV 太大，默认保存 meta 和 schema；特征表可选保存
    if args.save_features:
        X_out = pd.concat([meta_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        X_out.to_csv(train_feature_path, index=False, encoding="utf-8-sig")
        print(f"训练窗口特征表: {train_feature_path}")

    meta_df.to_csv(train_meta_path, index=False, encoding="utf-8-sig")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    splits, cv_name, n_splits = make_group_cv_splits(
        y=y,
        groups=groups,
        n_splits=args.cv,
        random_state=args.random_state,
    )

    print("\n========== 交叉验证设置 ==========")
    print(f"CV 方法: {cv_name}")
    print(f"折数: {n_splits}")
    print("说明：同一原始文件切出的所有窗口不会同时进入训练集和验证集。")

    k_list = parse_int_list(args.k_list)
    global_classes = sorted(np.unique(y).tolist())

    results = []
    best_score = -1.0
    best_std = 999.0
    best_k = None
    best_pred_df = None

    print("\n========== 开始 Raw-Window Flatten + XGBoost 训练 ==========")
    for k in k_list:
        k_eff = min(k, X.shape[1])
        print(f"\n正在评估：RawWindow_XGBoost_k{k_eff}")

        res, pred_df = evaluate_setting(
            X=X,
            y=y,
            groups=groups,
            source_files=source_files,
            k=k_eff,
            splits=splits,
            global_classes=global_classes,
            random_state=args.random_state,
        )

        print(
            f"  accuracy={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}, "
            f"macro-F1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}, "
            f"weighted-F1={res['weighted_f1_mean']:.4f}±{res['weighted_f1_std']:.4f}"
        )

        results.append(res)

        if (res["macro_f1_mean"] > best_score + 1e-12) or (
            abs(res["macro_f1_mean"] - best_score) <= 1e-12 and res["macro_f1_std"] < best_std
        ):
            best_score = res["macro_f1_mean"]
            best_std = res["macro_f1_std"]
            best_k = k_eff
            best_pred_df = pred_df.copy()
            print("  >>> 当前最优 Raw-Window XGBoost 已更新")

    results_df = pd.DataFrame(results).sort_values(
        by=["macro_f1_mean", "macro_f1_std"],
        ascending=[False, True],
    )

    cv_path = output_dir / "cv_results_raw_window_xgb.csv"
    results_df.to_csv(cv_path, index=False, encoding="utf-8-sig")

    print("\n========== Raw-Window XGBoost 结果 ==========")
    print(results_df.to_string(index=False))

    # 保存 OOF 预测
    oof_path = output_dir / "best_cv_file_predictions.csv"
    best_pred_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    y_true = best_pred_df["label"].to_numpy()
    y_pred = best_pred_df["predicted_label"].to_numpy()

    report_df = pd.DataFrame(
        classification_report(
            y_true, y_pred,
            labels=global_classes,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report_path = output_dir / "best_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=global_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in global_classes],
        columns=[f"pred_{c}" for c in global_classes],
    )
    cm_path = output_dir / "best_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print("\n========== 使用全部窗口样本训练最终 Raw-Window XGBoost ==========")
    print(f"最优 K: {best_k}")
    print(f"最优 macro-F1: {best_score:.4f} ± {best_std:.4f}")

    final_pipe = build_pipeline(k=best_k, random_state=args.random_state)
    final_pipe.fit(X, y)

    model_path = output_dir / "best_raw_window_xgb.pkl"
    artifact = {
        "pipeline": final_pipe,
        "model_name": "RawWindow_XGBoost",
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "best_macro_f1_std": float(best_std),
        "feature_names": feature_names,
        "class_labels": global_classes,
        "schema": schema,
        "cv_results": results,
        "aggregation": "mean_probability_by_group",
    }
    joblib.dump(artifact, model_path)

    info = {
        "model_name": "RawWindow_XGBoost",
        "window_len": int(args.window_len),
        "stride": int(args.stride),
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "best_macro_f1_std": float(best_std),
        "n_window_samples": int(X.shape[0]),
        "n_files": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "cv_method": cv_name,
        "n_splits": int(n_splits),
        "outputs": {
            "model": str(model_path),
            "cv_results": str(cv_path),
            "oof_file_predictions": str(oof_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(cm_path),
            "schema": str(schema_path),
        }
    }

    info_path = output_dir / "raw_window_xgb_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 训练完成 ==========")
    print(f"模型文件: {model_path}")
    print(f"CV 结果: {cv_path}")
    print(f"OOF 文件级预测: {oof_path}")
    print(f"分类报告: {report_path}")
    print(f"混淆矩阵: {cm_path}")
    print(f"信息文件: {info_path}")


def run_predict(args):
    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(args.model_file)
    pipe = artifact["pipeline"]
    schema = artifact["schema"]
    classes = artifact["class_labels"]

    print("\n========== Raw-Window XGBoost 测试集预测 ==========")
    print(f"模型: {artifact.get('model_name')}")
    print(f"K: {artifact.get('k')}")
    print(f"训练 CV macro-F1: {artifact.get('best_macro_f1'):.4f}")

    X_test, groups, source_files, feature_names, meta_df = build_test_raw_window_dataset(
        test_root=test_root,
        schema=schema,
        allow_short_last=args.allow_short_last,
    )

    print(f"测试窗口样本数: {X_test.shape[0]}")
    print(f"测试特征维度: {X_test.shape[1]}")
    print(f"测试文件数: {len(np.unique(groups))}")

    proba = pipe.predict_proba(X_test)
    proba = align_proba(proba, list(pipe.classes_), classes)

    prob_cols = [f"prob_class_{c}" for c in classes]
    window_pred = meta_df.copy()
    for j, c in enumerate(classes):
        window_pred[f"prob_class_{c}"] = proba[:, j]
    window_pred["predicted_label"] = [classes[i] for i in np.argmax(proba, axis=1)]

    file_pred = aggregate_to_file_level(
        proba=proba,
        groups=groups,
        source_files=source_files,
        classes=classes,
        y_window=None,
    )

    # 排序：如果文件名为数字.csv，则按数字排序
    def file_sort_key(x):
        stem = Path(str(x)).stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, str(x))

    file_pred = file_pred.sort_values(by="source_file", key=lambda s: s.map(file_sort_key)).reset_index(drop=True)

    file_output = output_dir / "raw_window_file_predictions.csv"
    window_output = output_dir / "raw_window_level_predictions.csv"
    submission_output = output_dir / "raw_window_submission.csv"
    info_output = output_dir / "raw_window_predict_info.json"

    file_pred.to_csv(file_output, index=False, encoding="utf-8-sig")
    window_pred.to_csv(window_output, index=False, encoding="utf-8-sig")

    # 提交文件：按常见格式保留 filename,label
    submission = file_pred[["source_file", "predicted_label"]].rename(
        columns={"source_file": "filename", "predicted_label": "label"}
    )
    submission.to_csv(submission_output, index=False, encoding="utf-8-sig")

    pred_dist = file_pred["predicted_label"].value_counts().sort_index().to_dict()
    low_conf = int((file_pred["top1_probability"] < 0.6).sum()) if "top1_probability" in file_pred.columns else None
    low_margin = int((file_pred["prob_margin"] < 0.1).sum()) if "prob_margin" in file_pred.columns else None

    info = {
        "model_file": str(args.model_file),
        "test_root": str(test_root),
        "n_test_files": int(len(file_pred)),
        "n_test_windows": int(len(window_pred)),
        "prediction_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "top1_probability_lt_0.6": low_conf,
        "prob_margin_lt_0.1": low_margin,
        "outputs": {
            "file_predictions": str(file_output),
            "window_predictions": str(window_output),
            "submission": str(submission_output),
        },
    }
    with open(info_output, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== 预测完成 ==========")
    print("预测类别分布:", pred_dist)
    print(f"top1_probability < 0.6: {low_conf}")
    print(f"prob_margin < 0.1: {low_margin}")
    print(f"文件级预测: {file_output}")
    print(f"窗口级预测: {window_output}")
    print(f"提交文件: {submission_output}")
    print(f"预测信息: {info_output}")


def main():
    parser = argparse.ArgumentParser(description="Raw-Window Flatten + XGBoost baseline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="训练 Raw-Window Flatten + XGBoost")
    p_train.add_argument("--train_root", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="artifacts_raw_window_xgb")
    p_train.add_argument("--window_len", type=int, default=120)
    p_train.add_argument("--stride", type=int, default=30)
    p_train.add_argument("--k_list", type=str, default="200,300,500,800")
    p_train.add_argument("--cv", type=int, default=5)
    p_train.add_argument("--random_state", type=int, default=42)
    p_train.add_argument("--allow_short_last", action="store_true")
    p_train.add_argument("--save_features", action="store_true", help="保存展开后的大特征表，可能较大。")
    p_train.set_defaults(func=run_train)

    p_pred = sub.add_parser("predict", help="预测测试集")
    p_pred.add_argument("--test_root", type=str, required=True)
    p_pred.add_argument("--model_file", type=str, required=True)
    p_pred.add_argument("--output_dir", type=str, default="test_results_raw_window")
    p_pred.add_argument("--allow_short_last", action="store_true")
    p_pred.set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
