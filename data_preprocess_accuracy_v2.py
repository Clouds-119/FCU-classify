"""
FCU 多传感器时间序列分类：精度提升版特征工程模块 v2

核心思想：
- 一个文件 = 一个样本；文件内部多行 = 该样本的传感器时间序列；
- 不把每一行拆成样本，而是提取更丰富的文件级统计、动态、控制逻辑和暖通物理特征；
- 训练阶段固定原始传感器列 schema，预测阶段复用同一套 schema，避免特征维度不一致。
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
TIME_COL_CANDIDATES = {
    "datetime", "date_time", "timestamp", "time", "date", "日期", "时间", "采集时间"
}

# 控制/离散状态列：提取状态占比、切换次数、最长持续段等
CONTROL_COLUMNS = [
    "FCU_CTRL", "FAN_CTRL", "FCU_CVLV_DM", "FCU_HVLV_DM", "FCU_DMPR_DM",
    "FCU_CVLV", "FCU_HVLV", "FCU_DMPR",
]

# 领域组合特征：差值/偏差
DOMAIN_DIFF_PAIRS = [
    ("RM_TEMP", "RMCLGSPT", "RM_TEMP_minus_CLGSPT"),
    ("RM_TEMP", "RMHTGSPT", "RM_TEMP_minus_HTGSPT"),
    ("RMCLGSPT", "RMHTGSPT", "CLGSPT_minus_HTGSPT"),
    ("FCU_DAT", "FCU_RAT", "DAT_minus_RAT"),
    ("FCU_MAT", "FCU_RAT", "MAT_minus_RAT"),
    ("FCU_DAT", "FCU_MAT", "DAT_minus_MAT"),
    ("FCU_CLG_EWT", "FCU_CLG_RWT", "CLG_EWT_minus_RWT"),
    ("FCU_HTG_EWT", "FCU_HTG_RWT", "HTG_EWT_minus_RWT"),
    ("FCU_CVLV", "FCU_CVLV_DM", "CVLV_minus_CVLV_DM"),
    ("FCU_HVLV", "FCU_HVLV_DM", "HVLV_minus_HVLV_DM"),
    ("FCU_DMPR", "FCU_DMPR_DM", "DMPR_minus_DMPR_DM"),
    ("FCU_OAT", "RM_TEMP", "OAT_minus_RM_TEMP"),
    ("FCU_OA_HUMD", "FCU_RA_HUMD", "OA_HUMD_minus_RA_HUMD"),
    ("FCU_DA_HUMD", "FCU_RA_HUMD", "DA_HUMD_minus_RA_HUMD"),
    ("FCU_MA_HUMD", "FCU_RA_HUMD", "MA_HUMD_minus_RA_HUMD"),
]

# 领域组合特征：比例
DOMAIN_RATIO_PAIRS = [
    ("FCU_DA_CFM", "FCU_WAT", "DA_CFM_div_WAT"),
    ("FCU_WAT", "FCU_DA_CFM", "WAT_div_DA_CFM"),
    ("FCU_OA_CFM", "FCU_DA_CFM", "OA_CFM_div_DA_CFM"),
    ("FCU_CLG_GPM", "FCU_DA_CFM", "CLG_GPM_div_DA_CFM"),
    ("FCU_HTG_GPM", "FCU_DA_CFM", "HTG_GPM_div_DA_CFM"),
]

# 近似换热强度特征，非严格物理量，只作为分类判别特征
DOMAIN_PRODUCT_TRIPLES = [
    ("FCU_CLG_GPM", "FCU_CLG_EWT", "FCU_CLG_RWT", "CLG_GPM_times_EWT_minus_RWT"),
    ("FCU_HTG_GPM", "FCU_HTG_EWT", "FCU_HTG_RWT", "HTG_GPM_times_EWT_minus_RWT"),
    ("FCU_DA_CFM", "FCU_RAT", "FCU_DAT", "DA_CFM_times_RAT_minus_DAT"),
]


def natural_key(path: Path | str):
    """让 2.csv 排在 10.csv 前面。"""
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_data_files(root: str | Path, recursive: bool = False) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"路径不存在：{root}\n当前工作目录：{Path.cwd()}\n"
            f"请检查 train_root/test_root 是否写成了真实数据目录。"
        )
    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=natural_key)


def read_table(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{file_path.suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_time_column(col: str) -> bool:
    c = str(col).strip().lower()
    return c in TIME_COL_CANDIDATES or "datetime" in c or "timestamp" in c


def get_time_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if is_time_column(c):
            return c
    return None


def infer_feature_columns(files: Sequence[str | Path], min_numeric_ratio: float = 0.5) -> List[str]:
    first_seen_order, seen = [], set()
    for f in files:
        df = read_table(f)
        if df.empty:
            continue
        for col in df.columns:
            if is_time_column(col):
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_ratio = float(s.notna().mean()) if len(s) else 0.0
            if numeric_ratio >= min_numeric_ratio and col not in seen:
                seen.add(col)
                first_seen_order.append(col)
    if not first_seen_order:
        raise ValueError("没有推断出任何数值传感器列，请检查数据文件和字段名。")
    return first_seen_order


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    n = max(len(df), 1)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series([np.nan] * n, dtype="float64")


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace(0, np.nan)
    out = a / denom
    return out.replace([np.inf, -np.inf], np.nan)


def _longest_run(mask: pd.Series) -> int:
    arr = mask.fillna(False).astype(bool).to_numpy()
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _trend_features(s: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    valid = s.dropna()
    if len(valid) < 3:
        return {
            f"{prefix}__slope": np.nan,
            f"{prefix}__trend_abs": np.nan,
            f"{prefix}__trend_r2": np.nan,
        }
    y = valid.to_numpy()
    x = np.arange(len(y), dtype="float64")
    try:
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    except Exception:
        slope, r2 = np.nan, np.nan
    return {
        f"{prefix}__slope": slope,
        f"{prefix}__trend_abs": abs(slope) if pd.notna(slope) else np.nan,
        f"{prefix}__trend_r2": r2,
    }


def _autocorr1(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").astype("float64").dropna()
    if len(s) < 4 or s.std() == 0:
        return np.nan
    return float(s.autocorr(lag=1))


def _series_stats(s: pd.Series, prefix: str, window_sizes: Sequence[int]) -> dict:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    diff = s.diff()
    q05 = s.quantile(0.05)
    q10 = s.quantile(0.10)
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    q90 = s.quantile(0.90)
    q95 = s.quantile(0.95)
    mean = s.mean()
    std = s.std()
    feats = {
        f"{prefix}__n": s.notna().sum(),
        f"{prefix}__missing_rate": s.isna().mean(),
        f"{prefix}__mean": mean,
        f"{prefix}__std": std,
        f"{prefix}__cv": std / (abs(mean) + 1e-6) if pd.notna(std) and pd.notna(mean) else np.nan,
        f"{prefix}__min": s.min(),
        f"{prefix}__max": s.max(),
        f"{prefix}__range": s.max() - s.min(),
        f"{prefix}__median": s.median(),
        f"{prefix}__q05": q05,
        f"{prefix}__q10": q10,
        f"{prefix}__q25": q25,
        f"{prefix}__q75": q75,
        f"{prefix}__q90": q90,
        f"{prefix}__q95": q95,
        f"{prefix}__iqr": q75 - q25,
        f"{prefix}__skew": s.skew(),
        f"{prefix}__kurt": s.kurt(),
        f"{prefix}__first": s.iloc[0] if len(s) else np.nan,
        f"{prefix}__last": s.iloc[-1] if len(s) else np.nan,
        f"{prefix}__last_minus_first": (s.iloc[-1] - s.iloc[0]) if len(s) else np.nan,
        f"{prefix}__diff_mean": diff.mean(),
        f"{prefix}__diff_std": diff.std(),
        f"{prefix}__diff_abs_mean": diff.abs().mean(),
        f"{prefix}__diff_abs_max": diff.abs().max(),
        f"{prefix}__diff_pos_ratio": (diff > 0).mean(),
        f"{prefix}__diff_neg_ratio": (diff < 0).mean(),
        f"{prefix}__zero_ratio": (s.abs() < 1e-8).mean(),
        f"{prefix}__nonzero_ratio": (s.abs() >= 1e-8).mean(),
        f"{prefix}__autocorr1": _autocorr1(s),
    }
    feats.update(_trend_features(s, prefix))

    # 多尺度滚动窗口特征，捕捉短期/中期波动
    for w in window_sizes:
        if w <= 1:
            continue
        roll = s.rolling(window=w, min_periods=max(2, min(w, 3)))
        r_mean = roll.mean()
        r_std = roll.std()
        r_rng = roll.max() - roll.min()
        feats[f"{prefix}__roll{w}_mean_mean"] = r_mean.mean()
        feats[f"{prefix}__roll{w}_mean_std"] = r_mean.std()
        feats[f"{prefix}__roll{w}_std_mean"] = r_std.mean()
        feats[f"{prefix}__roll{w}_std_max"] = r_std.max()
        feats[f"{prefix}__roll{w}_range_mean"] = r_rng.mean()
        feats[f"{prefix}__roll{w}_range_max"] = r_rng.max()
    return feats


def _control_features(s: pd.Series, prefix: str) -> dict:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    d = s.diff().abs()
    valid_n = max(int(s.notna().sum()), 1)
    feats = {
        f"{prefix}__switch_count": int((d > 1e-12).sum()),
        f"{prefix}__switch_rate": float((d > 1e-12).sum()) / valid_n,
        f"{prefix}__mean_abs_change": d.mean(),
        f"{prefix}__max_abs_change": d.max(),
        f"{prefix}__active_ratio": (s > 0).mean(),
        f"{prefix}__longest_active_run": _longest_run(s > 0),
    }
    # 常见状态占比，适合 FCU_CTRL=0/1/2、FAN_CTRL=1/2、阀门开度接近 0/1 等
    for val in [0, 1, 2]:
        feats[f"{prefix}__ratio_eq_{val}"] = (s.round(6) == val).mean()
        feats[f"{prefix}__longest_eq_{val}_run"] = _longest_run(s.round(6) == val)
    feats[f"{prefix}__ratio_low_0p1"] = (s <= 0.1).mean()
    feats[f"{prefix}__ratio_high_0p9"] = (s >= 0.9).mean()
    return feats


def _time_features(df: pd.DataFrame) -> dict:
    feats = {"meta__n_rows": len(df)}
    time_col = get_time_column(df)
    if time_col is None:
        return feats
    dt = pd.to_datetime(df[time_col], errors="coerce")
    valid = dt.dropna()
    if len(valid) >= 2:
        diffs = valid.sort_values().diff().dt.total_seconds().dropna()
        duration = (valid.max() - valid.min()).total_seconds()
        feats.update({
            "meta__duration_seconds": duration,
            "meta__sampling_seconds_mean": diffs.mean(),
            "meta__sampling_seconds_std": diffs.std(),
            "meta__start_hour": valid.iloc[0].hour,
            "meta__end_hour": valid.iloc[-1].hour,
            "meta__hour_mean": valid.dt.hour.mean(),
            "meta__weekend_ratio": (valid.dt.dayofweek >= 5).mean(),
        })
    return feats


def extract_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    window_sizes: Sequence[int] = (5, 15, 30),
) -> pd.Series:
    feats = {}
    feats.update(_time_features(df))

    # 单变量传感器特征
    for col in feature_columns:
        s = _safe_series(df, col)
        feats.update(_series_stats(s, col, window_sizes=window_sizes))
        if col in CONTROL_COLUMNS:
            feats.update(_control_features(s, col))

    # 领域差值/偏差特征
    for a, b, name in DOMAIN_DIFF_PAIRS:
        if a in feature_columns and b in feature_columns:
            diff = _safe_series(df, a) - _safe_series(df, b)
            feats.update(_series_stats(diff, name, window_sizes=window_sizes))

    # 领域比例特征
    for a, b, name in DOMAIN_RATIO_PAIRS:
        if a in feature_columns and b in feature_columns:
            ratio = _safe_ratio(_safe_series(df, a), _safe_series(df, b))
            feats.update(_series_stats(ratio, name, window_sizes=window_sizes))

    # 近似换热强度/负荷强度特征
    for flow_col, t_in_col, t_out_col, name in DOMAIN_PRODUCT_TRIPLES:
        if flow_col in feature_columns and t_in_col in feature_columns and t_out_col in feature_columns:
            val = _safe_series(df, flow_col) * (_safe_series(df, t_in_col) - _safe_series(df, t_out_col))
            feats.update(_series_stats(val, name, window_sizes=window_sizes))

    feat = pd.Series(feats, dtype="float64")
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def load_train_dataset(
    train_root: str | Path,
    feature_columns: Sequence[str] | None = None,
    window_sizes: Sequence[int] = (5, 15, 30),
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    train_root = Path(train_root)
    if not train_root.exists():
        raise FileNotFoundError(f"训练集路径不存在：{train_root}")

    label_dirs = [p for p in train_root.iterdir() if p.is_dir()]
    label_dirs = sorted(label_dirs, key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    if not label_dirs:
        raise FileNotFoundError(f"训练集目录下没有类别子文件夹：{train_root}")

    all_files, file_label_pairs = [], []
    for label_dir in label_dirs:
        files = list_data_files(label_dir, recursive=False)
        print(f"类别 {label_dir.name} 文件数量: {len(files)}")
        for f in files:
            all_files.append(f)
            label = int(label_dir.name) if label_dir.name.isdigit() else label_dir.name
            file_label_pairs.append((f, label))

    if not file_label_pairs:
        raise ValueError(f"没有读取到训练文件，请确认目录结构为：{train_root}/0/*.csv, {train_root}/1/*.csv ...")

    if feature_columns is None:
        feature_columns = infer_feature_columns(all_files)

    rows, labels, bad_files = [], [], []
    for f, label in file_label_pairs:
        try:
            df = read_table(f)
            if df.empty:
                warnings.warn(f"空文件已跳过：{f}")
                continue
            rows.append(extract_features(df, feature_columns, window_sizes=window_sizes))
            labels.append(label)
        except Exception as e:
            bad_files.append((str(f), str(e)))

    if bad_files:
        print("以下训练文件读取或特征提取失败，已跳过：")
        for f, e in bad_files[:10]:
            print(f"  - {f}: {e}")
        if len(bad_files) > 10:
            print(f"  ... 其余 {len(bad_files) - 10} 个略")

    if not rows:
        raise ValueError("训练特征为空：所有训练文件都未能成功提取特征。")

    X = pd.DataFrame(rows)
    y = np.asarray(labels)
    return X, y, list(feature_columns)


def load_test_dataset(
    test_root: str | Path,
    feature_columns: Sequence[str],
    window_sizes: Sequence[int] = (5, 15, 30),
) -> Tuple[pd.DataFrame, List[str]]:
    test_root = Path(test_root)
    files = list_data_files(test_root, recursive=False)
    if not files:
        raise ValueError(f"测试集目录没有 csv/xlsx/xls 文件：{test_root}")

    rows, filenames, bad_files = [], [], []
    for f in files:
        try:
            df = read_table(f)
            if df.empty:
                warnings.warn(f"空文件已跳过：{f}")
                continue
            rows.append(extract_features(df, feature_columns, window_sizes=window_sizes))
            filenames.append(f.name)
        except Exception as e:
            bad_files.append((str(f), str(e)))

    if bad_files:
        print("以下测试文件读取或特征提取失败，已跳过：")
        for f, e in bad_files[:10]:
            print(f"  - {f}: {e}")

    if not rows:
        raise ValueError("测试特征为空：所有测试文件都未能成功提取特征。")

    X = pd.DataFrame(rows)
    return X, filenames
