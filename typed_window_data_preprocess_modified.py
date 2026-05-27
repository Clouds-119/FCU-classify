# -*- coding: utf-8 -*-
"""
typed_window_data_preprocess.py

按变量类型区分的窗口级 FCU 特征工程脚本。

变量类型：
1. continuous 连续物理量：统计、波动、趋势、滚动窗口特征
2. binary 二值开关量：开启比例、关闭比例、切换次数、最长开启/关闭时间
3. multistate 多状态模式量：状态占比、主导状态、状态熵、切换次数
4. actuator 控制/执行器变量：开度统计、高/低/零开度比例、变化次数、持续时间
5. 指令-实际反馈组合：*_DM 与对应实际列的偏差、响应关系
6. 暖通物理组合：送回风温差、设定温度偏差、供回水温差等

输出与 window_train_only.py 兼容：
- window_train_features.csv
- window_test_features.csv
- window_schema.json

训练集处理：
python typed_window_data_preprocess.py train --train_root ".\\大作业2数据\\训练集" --output_dir "artifacts_window_typed" --window_len 240 --stride 60 --feature_windows 5,15,30

测试集处理：
python typed_window_data_preprocess.py test --test_root ".\\大作业2数据\\测试集" --schema_file "artifacts_window_typed\\window_schema.json" --output_dir "artifacts_window_typed"

后续训练：
python window_train_only.py --data_dir "artifacts_window_typed" --output_dir "artifacts_window_typed_train" --k_list 200,300,400,500 --cv 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from data_preprocess_accuracy_v2 import read_table

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


# ============================================================
# 1. 基础工具
# ============================================================

def parse_windows(s: str) -> Tuple[int, ...]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    vals = tuple(sorted(set(v for v in vals if v > 1)))
    return vals if vals else (5, 15, 30)


def list_data_files(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES],
        key=lambda p: p.name,
    )


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if any(key in c for key in ["time", "timestamp", "date", "datetime", "时间"]):
            return col
    return None


def sort_by_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    time_col = find_time_column(df)
    if time_col is None:
        return df.reset_index(drop=True)

    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.notna().sum() <= 1:
        return df.reset_index(drop=True)

    tmp = df.copy()
    tmp["_tmp_sort_time_"] = t
    tmp = tmp.sort_values("_tmp_sort_time_", kind="mergesort")
    tmp = tmp.drop(columns=["_tmp_sort_time_"])
    return tmp.reset_index(drop=True)


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    n = max(len(df), 1)
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series([np.nan] * n, dtype="float64")


def longest_run_bool(mask: pd.Series) -> int:
    arr = mask.fillna(False).astype(bool).to_numpy()
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def safe_entropy(values: pd.Series) -> float:
    v = values.dropna()
    if len(v) == 0:
        return np.nan
    counts = v.value_counts()
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def safe_autocorr1(s: pd.Series) -> float:
    x = s.dropna()
    if len(x) < 3:
        return np.nan
    if float(x.std(ddof=0)) == 0.0:
        return np.nan
    try:
        return float(x.autocorr(lag=1))
    except Exception:
        return np.nan


def safe_slope_and_r2(s: pd.Series) -> Tuple[float, float]:
    x = s.astype("float64").to_numpy()
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return np.nan, np.nan

    y = x[mask]
    t = np.arange(len(x), dtype="float64")[mask]

    if np.nanstd(y) == 0:
        return 0.0, 0.0

    try:
        coef = np.polyfit(t, y, 1)
        slope = float(coef[0])
        y_hat = coef[0] * t + coef[1]
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return slope, float(r2)
    except Exception:
        return np.nan, np.nan


def is_integer_like(s: pd.Series, tol: float = 1e-8) -> bool:
    x = s.dropna().to_numpy(dtype="float64")
    if len(x) == 0:
        return False
    return bool(np.all(np.abs(x - np.round(x)) < tol))


def normalize_thresholds(s: pd.Series) -> Tuple[float, float, float]:
    """根据变量尺度给出 low/high/change 阈值。"""
    x = s.dropna()
    if len(x) == 0:
        return 0.05, 0.8, 0.05
    vmax = float(np.nanmax(np.abs(x)))
    if vmax <= 1.5:
        return 0.05, 0.8, 0.05
    return 5.0, 80.0, 5.0


def sanitize_state_value(v) -> str:
    if pd.isna(v):
        return "nan"
    try:
        fv = float(v)
        if abs(fv - round(fv)) < 1e-8:
            return str(int(round(fv)))
        return str(fv).replace(".", "p").replace("-", "m")
    except Exception:
        return str(v).replace(".", "p").replace("-", "m").replace(" ", "_")


# ============================================================
# 2. 变量类型推断
# ============================================================

def infer_feature_columns_from_train(train_root: Path) -> List[str]:
    feature_cols = set()
    label_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for label_dir in label_dirs:
        for file_path in list_data_files(label_dir):
            try:
                df = read_table(file_path)
            except Exception as e:
                print(f"警告：读取失败，跳过 {file_path}: {e}")
                continue

            time_col = find_time_column(df)

            for col in df.columns:
                if col == time_col:
                    continue
                c_lower = str(col).lower()
                if c_lower in {"index", "idx", "unnamed: 0"}:
                    continue
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().sum() > 0:
                    feature_cols.add(str(col))

    feature_cols = sorted(feature_cols)
    if not feature_cols:
        raise RuntimeError("没有推断出任何可用原始变量列，请检查训练数据。")
    return feature_cols


def collect_column_profiles(train_root: Path, feature_columns: List[str], max_files_per_class: Optional[int] = None) -> Dict[str, dict]:
    values = {c: [] for c in feature_columns}
    label_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for label_dir in label_dirs:
        files = list_data_files(label_dir)
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        for fp in files:
            try:
                df = read_table(fp)
            except Exception:
                continue
            for col in feature_columns:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) > 0:
                        if len(s) > 500:
                            s = s.sample(500, random_state=42)
                        values[col].append(s)

    profiles = {}
    for col, parts in values.items():
        if parts:
            s_all = pd.concat(parts, ignore_index=True).astype("float64")
        else:
            s_all = pd.Series(dtype="float64")
        unique_vals = np.sort(s_all.dropna().unique())
        profiles[col] = {
            "n_valid": int(s_all.notna().sum()),
            "n_unique": int(len(unique_vals)),
            "unique_values_sample": [float(x) for x in unique_vals[:20]],
            "min": float(s_all.min()) if len(s_all) else None,
            "max": float(s_all.max()) if len(s_all) else None,
            "integer_like": bool(is_integer_like(s_all)),
        }
    return profiles


def infer_variable_type(col: str, profile: dict, manual_map: Optional[Dict[str, str]] = None) -> str:
    if manual_map and col in manual_map:
        return manual_map[col]

    name = col.lower()
    n_unique = profile.get("n_unique", 0)
    unique_vals = set(round(float(x), 8) for x in profile.get("unique_values_sample", []))
    integer_like = profile.get("integer_like", False)

    mode_keywords = ["mode", "state", "status", "sts", "gear", "speed_level", "stage", "档", "模式", "状态"]
    actuator_keywords = ["vlv", "valve", "cv", "hv", "dmpr", "damper", "开度", "阀", "风阀"]

    if n_unique <= 2 and unique_vals.issubset({0.0, 1.0}):
        return "binary"
    if any(k in name for k in actuator_keywords):
        return "actuator"
    if integer_like and n_unique <= 8 and any(k in name for k in mode_keywords):
        return "multistate"
    if integer_like and 2 < n_unique <= 6:
        return "multistate"
    return "continuous"


def get_default_domain_type_map() -> Dict[str, str]:
    """
    根据数据字段说明写入的领域默认变量类型。

    关键修正：
    - FAN_CTRL 是风机运行模式：1=Auto，2=Off，因此应作为 multistate，而不是 continuous。
    - FCU_CTRL 是 FCU 控制模式：0=Shutdown，1=Operate，2=Setback，因此作为 multistate。
    - FCU_SPD 是风机转速 rev/s，仍作为 continuous，但后续会额外构造风机一致性组合特征。
    - 阀门/风阀实际开度和控制信号均按 actuator 处理。
    """
    return {
        "FCU_CTRL": "multistate",
        "FAN_CTRL": "multistate",

        "FCU_CVLV": "actuator",
        "FCU_CVLV_DM": "actuator",
        "FCU_HVLV": "actuator",
        "FCU_HVLV_DM": "actuator",
        "FCU_DMPR": "actuator",
        "FCU_DMPR_DM": "actuator",

        "RM_TEMP": "continuous",
        "RMCLGSPT": "continuous",
        "RMHTGSPT": "continuous",
        "FCU_MAT": "continuous",
        "FCU_DAT": "continuous",
        "FCU_RAT": "continuous",
        "FCU_CLG_GPM": "continuous",
        "FCU_CLG_EWT": "continuous",
        "FCU_CLG_RWT": "continuous",
        "FCU_HTG_GPM": "continuous",
        "FCU_HTG_EWT": "continuous",
        "FCU_HTG_RWT": "continuous",
        "FCU_DA_CFM": "continuous",
        "FCU_OA_CFM": "continuous",
        "FCU_SPD": "continuous",
        "FCU_OAT": "continuous",
        "FCU_WAT": "continuous",
        "FCU_MA_HUMD": "continuous",
        "FCU_OA_HUMD": "continuous",
        "FCU_DA_HUMD": "continuous",
        "FCU_RA_HUMD": "continuous",
    }


# ============================================================
# 3. 类型化特征提取
# ============================================================

def continuous_features(s: pd.Series, prefix: str, rolling_windows: Tuple[int, ...]) -> Dict[str, float]:
    feats = {}
    x = s.astype("float64")
    valid = x.dropna()
    feats[f"{prefix}__n_valid"] = int(valid.shape[0])
    feats[f"{prefix}__nan_ratio"] = float(x.isna().mean())

    if len(valid) == 0:
        keys = ["mean", "std", "min", "max", "median", "q25", "q75", "range", "iqr",
                "diff_mean", "diff_std", "diff_abs_mean", "diff_abs_max",
                "last_minus_first", "slope", "trend_r2", "autocorr1"]
        for k in keys:
            feats[f"{prefix}__{k}"] = np.nan
        return feats

    feats[f"{prefix}__mean"] = float(valid.mean())
    feats[f"{prefix}__std"] = float(valid.std())
    feats[f"{prefix}__min"] = float(valid.min())
    feats[f"{prefix}__max"] = float(valid.max())
    feats[f"{prefix}__median"] = float(valid.median())
    feats[f"{prefix}__q25"] = float(valid.quantile(0.25))
    feats[f"{prefix}__q75"] = float(valid.quantile(0.75))
    feats[f"{prefix}__range"] = float(valid.max() - valid.min())
    feats[f"{prefix}__iqr"] = float(valid.quantile(0.75) - valid.quantile(0.25))

    diff = x.diff().dropna()
    if len(diff) > 0:
        feats[f"{prefix}__diff_mean"] = float(diff.mean())
        feats[f"{prefix}__diff_std"] = float(diff.std())
        feats[f"{prefix}__diff_abs_mean"] = float(diff.abs().mean())
        feats[f"{prefix}__diff_abs_max"] = float(diff.abs().max())
        feats[f"{prefix}__diff_pos_ratio"] = float((diff > 0).mean())
        feats[f"{prefix}__diff_neg_ratio"] = float((diff < 0).mean())
    else:
        for k in ["diff_mean", "diff_std", "diff_abs_mean", "diff_abs_max", "diff_pos_ratio", "diff_neg_ratio"]:
            feats[f"{prefix}__{k}"] = np.nan

    feats[f"{prefix}__first"] = float(valid.iloc[0]) if len(valid) >= 1 else np.nan
    feats[f"{prefix}__last"] = float(valid.iloc[-1]) if len(valid) >= 1 else np.nan
    feats[f"{prefix}__last_minus_first"] = float(valid.iloc[-1] - valid.iloc[0]) if len(valid) >= 2 else np.nan

    slope, r2 = safe_slope_and_r2(x)
    feats[f"{prefix}__slope"] = slope
    feats[f"{prefix}__trend_r2"] = r2
    feats[f"{prefix}__autocorr1"] = safe_autocorr1(x)

    for w in rolling_windows:
        if len(x) >= w:
            roll_mean = x.rolling(w, min_periods=max(2, w // 2)).mean()
            roll_std = x.rolling(w, min_periods=max(2, w // 2)).std()
            roll_range = x.rolling(w, min_periods=max(2, w // 2)).max() - x.rolling(w, min_periods=max(2, w // 2)).min()
            feats[f"{prefix}__roll{w}_mean_mean"] = float(roll_mean.mean())
            feats[f"{prefix}__roll{w}_mean_std"] = float(roll_mean.std())
            feats[f"{prefix}__roll{w}_std_mean"] = float(roll_std.mean())
            feats[f"{prefix}__roll{w}_std_max"] = float(roll_std.max())
            feats[f"{prefix}__roll{w}_range_mean"] = float(roll_range.mean())
            feats[f"{prefix}__roll{w}_range_max"] = float(roll_range.max())
        else:
            for k in ["mean_mean", "mean_std", "std_mean", "std_max", "range_mean", "range_max"]:
                feats[f"{prefix}__roll{w}_{k}"] = np.nan
    return feats


def binary_features(s: pd.Series, prefix: str) -> Dict[str, float]:
    feats = {}
    x = s.astype("float64")
    valid = x.dropna()
    feats[f"{prefix}__n_valid"] = int(valid.shape[0])
    feats[f"{prefix}__nan_ratio"] = float(x.isna().mean())
    if len(valid) == 0:
        for k in ["on_ratio", "off_ratio", "switch_count", "switch_rate", "rise_count", "fall_count",
                  "longest_on_run", "longest_off_run", "first_state", "last_state"]:
            feats[f"{prefix}__{k}"] = np.nan
        return feats

    xb = (x >= 0.5)
    feats[f"{prefix}__on_ratio"] = float(xb.mean())
    feats[f"{prefix}__off_ratio"] = float((~xb).mean())
    arr = xb.dropna().astype(int)
    diff = arr.diff().dropna()
    switch_count = int((diff != 0).sum()) if len(diff) else 0
    feats[f"{prefix}__switch_count"] = switch_count
    feats[f"{prefix}__switch_rate"] = float(switch_count / max(len(arr) - 1, 1))
    feats[f"{prefix}__rise_count"] = int((diff > 0).sum()) if len(diff) else 0
    feats[f"{prefix}__fall_count"] = int((diff < 0).sum()) if len(diff) else 0
    feats[f"{prefix}__longest_on_run"] = longest_run_bool(xb)
    feats[f"{prefix}__longest_off_run"] = longest_run_bool(~xb)
    feats[f"{prefix}__first_state"] = float(arr.iloc[0]) if len(arr) else np.nan
    feats[f"{prefix}__last_state"] = float(arr.iloc[-1]) if len(arr) else np.nan
    return feats


def multistate_features(s: pd.Series, prefix: str, max_states: int = 8) -> Dict[str, float]:
    feats = {}
    x = s.astype("float64")
    valid = x.dropna()
    feats[f"{prefix}__n_valid"] = int(valid.shape[0])
    feats[f"{prefix}__nan_ratio"] = float(x.isna().mean())
    if len(valid) == 0:
        for k in ["n_states", "state_entropy", "dominant_state", "dominant_ratio", "switch_count", "switch_rate"]:
            feats[f"{prefix}__{k}"] = np.nan
        return feats

    rounded = valid.round(0)
    counts = rounded.value_counts()
    n = len(rounded)
    feats[f"{prefix}__n_states"] = int(counts.shape[0])
    feats[f"{prefix}__state_entropy"] = safe_entropy(rounded)
    feats[f"{prefix}__dominant_state"] = float(counts.index[0])
    feats[f"{prefix}__dominant_ratio"] = float(counts.iloc[0] / n)
    arr = rounded.astype(int)
    diff = arr.diff().dropna()
    switch_count = int((diff != 0).sum()) if len(diff) else 0
    feats[f"{prefix}__switch_count"] = switch_count
    feats[f"{prefix}__switch_rate"] = float(switch_count / max(len(arr) - 1, 1))
    full = x.round(0)
    for st in sorted(counts.index.tolist())[:max_states]:
        st_name = sanitize_state_value(st)
        mask = (full == st)
        feats[f"{prefix}__state_{st_name}_ratio"] = float(mask.mean())
        feats[f"{prefix}__state_{st_name}_longest_run"] = longest_run_bool(mask)
    return feats


def actuator_features(s: pd.Series, prefix: str, rolling_windows: Tuple[int, ...]) -> Dict[str, float]:
    feats = {}
    x = s.astype("float64")
    feats.update(continuous_features(x, prefix, rolling_windows=()))
    low_th, high_th, change_th = normalize_thresholds(x)
    valid = x.dropna()
    if len(valid) == 0:
        for k in ["zero_ratio", "low_open_ratio", "high_open_ratio", "mid_open_ratio",
                  "change_count", "change_rate", "large_change_count", "longest_zero_run", "longest_high_open_run"]:
            feats[f"{prefix}__{k}"] = np.nan
        return feats
    zero_mask = x.abs() <= low_th
    high_mask = x >= high_th
    low_mask = x <= low_th
    mid_mask = (~low_mask) & (~high_mask) & x.notna()
    feats[f"{prefix}__zero_ratio"] = float(zero_mask.mean())
    feats[f"{prefix}__low_open_ratio"] = float(low_mask.mean())
    feats[f"{prefix}__high_open_ratio"] = float(high_mask.mean())
    feats[f"{prefix}__mid_open_ratio"] = float(mid_mask.mean())
    diff = x.diff().abs().dropna()
    feats[f"{prefix}__change_count"] = int((diff > change_th).sum()) if len(diff) else 0
    feats[f"{prefix}__change_rate"] = float((diff > change_th).mean()) if len(diff) else np.nan
    feats[f"{prefix}__large_change_count"] = int((diff > 2 * change_th).sum()) if len(diff) else 0
    feats[f"{prefix}__longest_zero_run"] = longest_run_bool(zero_mask)
    feats[f"{prefix}__longest_high_open_run"] = longest_run_bool(high_mask)
    return feats


def pair_error_features(actual: pd.Series, cmd: pd.Series, prefix: str) -> Dict[str, float]:
    feats = {}
    a = actual.astype("float64")
    c = cmd.astype("float64")
    err = a - c
    abs_err = err.abs()
    feats[f"{prefix}__err_mean"] = float(err.mean()) if err.notna().any() else np.nan
    feats[f"{prefix}__err_std"] = float(err.std()) if err.notna().any() else np.nan
    feats[f"{prefix}__abs_err_mean"] = float(abs_err.mean()) if abs_err.notna().any() else np.nan
    feats[f"{prefix}__abs_err_max"] = float(abs_err.max()) if abs_err.notna().any() else np.nan
    feats[f"{prefix}__abs_err_q90"] = float(abs_err.quantile(0.9)) if abs_err.notna().any() else np.nan
    _, _, change_th_a = normalize_thresholds(a)
    _, _, change_th_c = normalize_thresholds(c)
    err_th = max(change_th_a, change_th_c)
    feats[f"{prefix}__within_tol_ratio"] = float((abs_err <= err_th).mean()) if abs_err.notna().any() else np.nan
    feats[f"{prefix}__large_err_ratio"] = float((abs_err > 2 * err_th).mean()) if abs_err.notna().any() else np.nan
    cmd_change = c.diff().abs() > change_th_c
    actual_change = a.diff().abs() > change_th_a
    feats[f"{prefix}__cmd_change_count"] = int(cmd_change.fillna(False).sum())
    feats[f"{prefix}__actual_change_count"] = int(actual_change.fillna(False).sum())
    cmd_change_count = int(cmd_change.fillna(False).sum())
    if cmd_change_count > 0:
        no_resp = cmd_change.fillna(False) & (~actual_change.fillna(False))
        feats[f"{prefix}__cmd_change_no_actual_change_ratio"] = float(no_resp.sum() / cmd_change_count)
    else:
        feats[f"{prefix}__cmd_change_no_actual_change_ratio"] = np.nan
    cmd_diff = c.diff()
    act_diff = a.diff()
    for lag in [1, 2, 3]:
        try:
            corr = cmd_diff.corr(act_diff.shift(-lag))
            feats[f"{prefix}__response_corr_lag{lag}"] = float(corr) if pd.notna(corr) else np.nan
        except Exception:
            feats[f"{prefix}__response_corr_lag{lag}"] = np.nan
    return feats


def physical_pair_features(df: pd.DataFrame) -> Dict[str, float]:
    pairs = [
        ("RM_TEMP", "RMCLGSPT", "RM_TEMP_minus_RMCLGSPT"),
        ("RM_TEMP", "RMHTGSPT", "RM_TEMP_minus_RMHTGSPT"),
        ("FCU_DAT", "FCU_RAT", "FCU_DAT_minus_FCU_RAT"),
        ("FCU_CLG_EWT", "FCU_CLG_RWT", "FCU_CLG_EWT_minus_FCU_CLG_RWT"),
        ("FCU_HTG_EWT", "FCU_HTG_RWT", "FCU_HTG_EWT_minus_FCU_HTG_RWT"),
        ("FCU_OAT", "RM_TEMP", "FCU_OAT_minus_RM_TEMP"),
    ]
    feats = {}
    for a, b, name in pairs:
        if a in df.columns and b in df.columns:
            s = safe_series(df, a) - safe_series(df, b)
            feats.update(continuous_features(s, name, rolling_windows=()))
    return feats


def command_actual_pair_features(df: pd.DataFrame) -> Dict[str, float]:
    feats = {}
    cols = set(df.columns)
    for col in list(cols):
        if str(col).endswith("_DM"):
            base = str(col)[:-3]
            if base in cols:
                actual = safe_series(df, base)
                cmd = safe_series(df, col)
                feats.update(pair_error_features(actual, cmd, f"{base}_actual_minus_cmd"))
    return feats


def _positive_mask(s: pd.Series, ratio: float = 0.01, min_th: float = 1e-8) -> pd.Series:
    """
    针对转速、风量、功耗等非负物理量，自动给出“非零/有效运行”判断。
    阈值不直接写死为 0，是为了避免极小噪声被当成有效运行。
    """
    x = s.astype("float64")
    valid = x.dropna()
    if len(valid) == 0:
        return pd.Series([False] * len(x), index=x.index)

    vmax = float(np.nanmax(np.abs(valid)))
    if vmax <= 0:
        th = min_th
    elif vmax <= 1.5:
        th = max(0.05, min_th)
    else:
        th = max(vmax * ratio, min_th)

    return x > th


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """
    安全相关系数：当任一变量缺失太多或恒定时返回 NaN，避免 numpy 的 invalid divide 警告。
    """
    aa = a.astype("float64")
    bb = b.astype("float64")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 3:
        return np.nan
    aa = aa[mask]
    bb = bb[mask]
    if float(aa.std(ddof=0)) == 0.0 or float(bb.std(ddof=0)) == 0.0:
        return np.nan
    try:
        return float(aa.corr(bb))
    except Exception:
        return np.nan


def _conditional_mean(value: pd.Series, condition: pd.Series) -> float:
    mask = condition.fillna(False).astype(bool) & value.notna()
    if mask.sum() == 0:
        return np.nan
    return float(value[mask].mean())


def _conditional_ratio(condition: pd.Series, base: pd.Series) -> float:
    base_mask = base.fillna(False).astype(bool)
    denom = int(base_mask.sum())
    if denom == 0:
        return np.nan
    return float((condition.fillna(False).astype(bool) & base_mask).sum() / denom)


def fan_consistency_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    风机运行一致性特征。

    依据字段说明：
    - FAN_CTRL: 风机运行模式，1=Auto，2=Off
    - FCU_CTRL: FCU 控制模式，0=Shutdown，1=Operate，2=Setback
    - FCU_SPD: 风机转速 rev/s
    - FCU_DA_CFM: 送风风量 CFM
    - FCU_WAT: 风机功耗 Watt

    这类特征比单独看 FCU_SPD 均值更有意义：
    - Auto/Operate 时转速是否为 0
    - Off/Shutdown 时转速是否非 0
    - 转速、风量、功耗三者是否一致
    """
    feats: Dict[str, float] = {}

    required_any = {"FAN_CTRL", "FCU_CTRL", "FCU_SPD", "FCU_DA_CFM", "FCU_WAT"}
    if not any(c in df.columns for c in required_any):
        return feats

    fan_ctrl = safe_series(df, "FAN_CTRL")
    fcu_ctrl = safe_series(df, "FCU_CTRL")
    spd = safe_series(df, "FCU_SPD")
    da_cfm = safe_series(df, "FCU_DA_CFM")
    wat = safe_series(df, "FCU_WAT")

    fan_auto = fan_ctrl.round(0) == 1
    fan_off = fan_ctrl.round(0) == 2

    fcu_shutdown = fcu_ctrl.round(0) == 0
    fcu_operate = fcu_ctrl.round(0) == 1
    fcu_setback = fcu_ctrl.round(0) == 2

    spd_on = _positive_mask(spd)
    air_on = _positive_mask(da_cfm)
    power_on = _positive_mask(wat)

    n = max(len(df), 1)

    # FAN_CTRL / FCU_CTRL 状态比例
    feats["fan_consistency__fan_auto_ratio"] = float(fan_auto.mean())
    feats["fan_consistency__fan_off_ratio"] = float(fan_off.mean())
    feats["fan_consistency__fcu_shutdown_ratio"] = float(fcu_shutdown.mean())
    feats["fan_consistency__fcu_operate_ratio"] = float(fcu_operate.mean())
    feats["fan_consistency__fcu_setback_ratio"] = float(fcu_setback.mean())

    # 转速自身运行特征
    feats["fan_consistency__speed_nonzero_ratio"] = float(spd_on.mean())
    feats["fan_consistency__speed_zero_ratio"] = float((~spd_on).mean())
    feats["fan_consistency__longest_speed_zero_run"] = longest_run_bool(~spd_on)
    feats["fan_consistency__longest_speed_nonzero_run"] = longest_run_bool(spd_on)

    # 模式-转速一致性
    feats["fan_consistency__speed_when_auto_mean"] = _conditional_mean(spd, fan_auto)
    feats["fan_consistency__speed_when_operate_mean"] = _conditional_mean(spd, fcu_operate)

    feats["fan_consistency__auto_but_speed_zero_ratio"] = _conditional_ratio(~spd_on, fan_auto)
    feats["fan_consistency__off_but_speed_nonzero_ratio"] = _conditional_ratio(spd_on, fan_off)
    feats["fan_consistency__operate_but_speed_zero_ratio"] = _conditional_ratio(~spd_on, fcu_operate)
    feats["fan_consistency__shutdown_but_speed_nonzero_ratio"] = _conditional_ratio(spd_on, fcu_shutdown)

    # 转速-风量-功耗一致性
    feats["fan_consistency__airflow_nonzero_ratio"] = float(air_on.mean())
    feats["fan_consistency__power_nonzero_ratio"] = float(power_on.mean())

    feats["fan_consistency__speed_airflow_corr"] = _safe_corr(spd, da_cfm)
    feats["fan_consistency__speed_power_corr"] = _safe_corr(spd, wat)
    feats["fan_consistency__airflow_power_corr"] = _safe_corr(da_cfm, wat)

    feats["fan_consistency__speed_nonzero_but_airflow_zero_ratio"] = float((spd_on & (~air_on)).sum() / n)
    feats["fan_consistency__airflow_nonzero_but_speed_zero_ratio"] = float((air_on & (~spd_on)).sum() / n)
    feats["fan_consistency__speed_nonzero_but_power_zero_ratio"] = float((spd_on & (~power_on)).sum() / n)
    feats["fan_consistency__power_nonzero_but_speed_zero_ratio"] = float((power_on & (~spd_on)).sum() / n)

    # 简单效率/比例特征：只在分母有效时计算
    mask_spd = spd_on & spd.notna() & da_cfm.notna()
    if mask_spd.sum() > 0:
        feats["fan_consistency__airflow_per_speed_mean"] = float((da_cfm[mask_spd] / (spd[mask_spd].abs() + 1e-6)).mean())
    else:
        feats["fan_consistency__airflow_per_speed_mean"] = np.nan

    mask_power = power_on & wat.notna() & da_cfm.notna()
    if mask_power.sum() > 0:
        feats["fan_consistency__airflow_per_power_mean"] = float((da_cfm[mask_power] / (wat[mask_power].abs() + 1e-6)).mean())
    else:
        feats["fan_consistency__airflow_per_power_mean"] = np.nan

    return feats


def typed_extract_features(df: pd.DataFrame, feature_columns: List[str], type_map: Dict[str, str], rolling_windows: Tuple[int, ...]) -> Dict[str, float]:
    feats = {}
    for col in feature_columns:
        s = safe_series(df, col)
        var_type = type_map.get(col, "continuous")
        if var_type == "binary":
            feats.update(binary_features(s, col))
        elif var_type == "multistate":
            feats.update(multistate_features(s, col))
        elif var_type == "actuator":
            feats.update(actuator_features(s, col, rolling_windows=rolling_windows))
        else:
            feats.update(continuous_features(s, col, rolling_windows=rolling_windows))
    feats.update(physical_pair_features(df))
    feats.update(command_actual_pair_features(df))
    feats.update(fan_consistency_features(df))
    return feats


# ============================================================
# 4. 窗口切分与文件处理
# ============================================================

def make_window_slices(n_rows: int, window_len: int, stride: int, allow_short_last: bool = False, min_short_ratio: float = 0.8) -> List[Tuple[int, int]]:
    if n_rows <= 0:
        return []
    if window_len <= 0:
        raise ValueError("window_len 必须大于 0")
    if stride <= 0:
        raise ValueError("stride 必须大于 0")
    if n_rows < window_len:
        if n_rows >= max(1, int(window_len * min_short_ratio)):
            return [(0, n_rows)]
        return []
    slices = []
    start = 0
    while start + window_len <= n_rows:
        slices.append((start, start + window_len))
        start += stride
    if allow_short_last and start < n_rows:
        short_len = n_rows - start
        if short_len >= int(window_len * min_short_ratio):
            slices.append((start, n_rows))
    return slices


def extract_window_features_for_file(file_path: Path, feature_columns: List[str], type_map: Dict[str, str], feature_windows: Tuple[int, ...], window_len: int, stride: int, label: Optional[int] = None, allow_short_last: bool = False) -> List[Dict]:
    df = read_table(file_path)
    df = sort_by_time_if_possible(df)
    slices = make_window_slices(len(df), window_len, stride, allow_short_last=allow_short_last)
    rows = []
    for win_id, (start, end) in enumerate(slices):
        wdf = df.iloc[start:end].reset_index(drop=True)
        feats = typed_extract_features(wdf, feature_columns=feature_columns, type_map=type_map, rolling_windows=feature_windows)
        row = {
            "source_file": file_path.name,
            "group_id": file_path.stem,
            "window_id": win_id,
            "start_idx": start,
            "end_idx": end,
            "window_n_rows": end - start,
        }
        if label is not None:
            row["label"] = int(label)
        row.update(feats)
        rows.append(row)
    return rows


# ============================================================
# 5. train/test 数据生成
# ============================================================

def load_manual_type_map(config_file: Optional[str]) -> Dict[str, str]:
    if not config_file:
        return {}
    p = Path(config_file)
    if not p.exists():
        raise FileNotFoundError(f"变量类型配置文件不存在：{p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if all(isinstance(v, str) for v in data.values()):
        return {str(k): str(v) for k, v in data.items()}
    out = {}
    for typ, cols in data.items():
        if isinstance(cols, list):
            for c in cols:
                out[str(c)] = str(typ)
    return out


def build_type_map(train_root: Path, feature_columns: List[str], manual_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, dict]]:
    profiles = collect_column_profiles(train_root, feature_columns)
    type_map = {}
    for col in feature_columns:
        type_map[col] = infer_variable_type(col, profiles[col], manual_map=manual_map)
    return type_map, profiles


def build_train_dataset(train_root: Path, output_dir: Path, window_len: int, stride: int, feature_windows: Tuple[int, ...], allow_short_last: bool, type_config: Optional[str]) -> None:
    if not train_root.exists():
        raise FileNotFoundError(f"训练集路径不存在：{train_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== 推断原始变量列 ==========")
    feature_columns = infer_feature_columns_from_train(train_root)
    print(f"原始变量列数: {len(feature_columns)}")
    print(f"前 10 个变量列: {feature_columns[:10]}")

    # 先加载领域默认类型，再用用户手动配置覆盖。
    # 这样即使不提供 type_config，也会把 FAN_CTRL 正确作为 multistate，
    # 并保持 FCU_SPD 为 continuous。
    manual_map = get_default_domain_type_map()
    manual_map.update(load_manual_type_map(type_config))
    print("\n========== 判断变量类型 ==========")
    type_map, profiles = build_type_map(train_root, feature_columns, manual_map)
    print("变量类型分布:", dict(Counter(type_map.values())))

    type_df = pd.DataFrame([
        {
            "column": col,
            "type": type_map[col],
            "n_unique": profiles[col]["n_unique"],
            "min": profiles[col]["min"],
            "max": profiles[col]["max"],
            "integer_like": profiles[col]["integer_like"],
            "unique_values_sample": profiles[col]["unique_values_sample"],
        }
        for col in feature_columns
    ])
    type_df.to_csv(output_dir / "feature_type_map.csv", index=False, encoding="utf-8-sig")

    label_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    all_rows = []
    file_summary_rows = []

    print("\n========== 开始生成类型化窗口训练特征 ==========")
    for label_dir in label_dirs:
        try:
            label = int(label_dir.name)
        except ValueError:
            print(f"警告：类别文件夹名称不是整数，跳过：{label_dir}")
            continue
        files = list_data_files(label_dir)
        print(f"类别 {label} 文件数量: {len(files)}")
        for fp in files:
            try:
                rows = extract_window_features_for_file(fp, feature_columns, type_map, feature_windows, window_len, stride, label=label, allow_short_last=allow_short_last)
            except Exception as e:
                print(f"警告：处理失败，跳过 {fp}: {e}")
                continue
            all_rows.extend(rows)
            file_summary_rows.append({"source_file": fp.name, "group_id": fp.stem, "label": label, "n_windows": len(rows)})

    if not all_rows:
        raise RuntimeError("没有生成任何窗口样本，请检查窗口长度、步长和数据文件。")

    df_win = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan)
    df_summary = pd.DataFrame(file_summary_rows)
    meta_cols = ["source_file", "group_id", "window_id", "start_idx", "end_idx", "window_n_rows", "label"]
    feature_names = [c for c in df_win.columns if c not in meta_cols]
    all_nan_features = [c for c in feature_names if df_win[c].isna().all()]
    if all_nan_features:
        print(f"删除全 NaN 特征 {len(all_nan_features)} 个，例如：{all_nan_features[:10]}")
        df_win = df_win.drop(columns=all_nan_features)
        feature_names = [c for c in feature_names if c not in all_nan_features]

    train_csv = output_dir / "window_train_features.csv"
    summary_csv = output_dir / "window_train_file_summary.csv"
    schema_json = output_dir / "window_schema.json"
    df_win.to_csv(train_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    schema = {
        "task": "typed_window_level_fcu_preprocess",
        "train_root": str(train_root),
        "window_len": int(window_len),
        "stride": int(stride),
        "feature_windows": list(feature_windows),
        "allow_short_last": bool(allow_short_last),
        "feature_columns": feature_columns,
        "type_map": type_map,
        "meta_cols": meta_cols,
        "feature_names": feature_names,
        "label_col": "label",
        "group_col": "group_id",
        "n_files": int(df_summary.shape[0]),
        "n_window_samples": int(df_win.shape[0]),
        "n_features": int(len(feature_names)),
        "label_distribution_files": {str(k): int(v) for k, v in Counter(df_summary["label"]).items()},
        "label_distribution_windows": {str(k): int(v) for k, v in Counter(df_win["label"]).items()},
        "all_nan_features_removed": all_nan_features,
    }
    with open(schema_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print("\n========== 类型化窗口训练数据生成完成 ==========")
    print(f"文件级样本数: {df_summary.shape[0]}")
    print(f"窗口级样本数: {df_win.shape[0]}")
    print(f"窗口特征维度: {len(feature_names)}")
    print("文件级类别分布:", dict(sorted(Counter(df_summary["label"]).items())))
    print("窗口级类别分布:", dict(sorted(Counter(df_win["label"]).items())))
    print(f"变量类型表: {output_dir / 'feature_type_map.csv'}")
    print(f"训练窗口特征表: {train_csv}")
    print(f"文件窗口统计表: {summary_csv}")
    print(f"schema 文件: {schema_json}")


def build_test_dataset(test_root: Path, schema_file: Path, output_dir: Path, window_len: Optional[int] = None, stride: Optional[int] = None, feature_windows: Optional[Tuple[int, ...]] = None, allow_short_last: Optional[bool] = None) -> None:
    if not test_root.exists():
        raise FileNotFoundError(f"测试集路径不存在：{test_root}")
    if not schema_file.exists():
        raise FileNotFoundError(f"schema 文件不存在：{schema_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_columns = schema["feature_columns"]
    type_map = schema["type_map"]
    train_feature_names = schema["feature_names"]
    window_len = int(window_len if window_len is not None else schema["window_len"])
    stride = int(stride if stride is not None else schema["stride"])
    feature_windows = tuple(feature_windows if feature_windows is not None else schema["feature_windows"])
    allow_short_last = bool(allow_short_last if allow_short_last is not None else schema.get("allow_short_last", False))

    files = list_data_files(test_root)
    if not files:
        raise RuntimeError(f"测试集目录下没有支持格式的数据文件：{test_root}")

    all_rows = []
    file_summary_rows = []
    print("\n========== 开始生成类型化窗口测试特征 ==========")
    print(f"测试文件数量: {len(files)}")
    print(f"窗口长度: {window_len}, 步长: {stride}, 特征滚动窗口: {feature_windows}")
    for fp in files:
        try:
            rows = extract_window_features_for_file(fp, feature_columns, type_map, feature_windows, window_len, stride, label=None, allow_short_last=allow_short_last)
        except Exception as e:
            print(f"警告：处理失败，跳过 {fp}: {e}")
            continue
        all_rows.extend(rows)
        file_summary_rows.append({"source_file": fp.name, "group_id": fp.stem, "n_windows": len(rows)})

    if not all_rows:
        raise RuntimeError("没有生成任何测试窗口样本，请检查窗口长度、步长和测试数据。")

    df_win = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan)
    df_summary = pd.DataFrame(file_summary_rows)
    meta_cols = ["source_file", "group_id", "window_id", "start_idx", "end_idx", "window_n_rows"]
    for c in meta_cols:
        if c not in df_win.columns:
            df_win[c] = np.nan
    feature_part = df_win.drop(columns=[c for c in meta_cols if c in df_win.columns], errors="ignore")
    feature_part = feature_part.reindex(columns=train_feature_names)
    df_aligned = pd.concat([df_win[meta_cols], feature_part], axis=1)
    test_csv = output_dir / "window_test_features.csv"
    summary_csv = output_dir / "window_test_file_summary.csv"
    df_aligned.to_csv(test_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n========== 类型化窗口测试数据生成完成 ==========")
    print(f"测试文件数: {df_summary.shape[0]}")
    print(f"测试窗口样本数: {df_aligned.shape[0]}")
    print(f"测试窗口特征维度: {len(train_feature_names)}")
    print(f"测试窗口特征表: {test_csv}")
    print(f"测试文件窗口统计表: {summary_csv}")


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="按变量类型区分的窗口级 FCU 特征工程脚本")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train", help="生成训练集类型化窗口特征")
    p_train.add_argument("--train_root", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="artifacts_window_typed")
    p_train.add_argument("--window_len", type=int, default=240)
    p_train.add_argument("--stride", type=int, default=60)
    p_train.add_argument("--feature_windows", type=str, default="5,15,30")
    p_train.add_argument("--allow_short_last", action="store_true")
    p_train.add_argument("--type_config", type=str, default=None, help="可选，变量类型手动配置 JSON")

    p_test = subparsers.add_parser("test", help="生成测试集类型化窗口特征")
    p_test.add_argument("--test_root", type=str, required=True)
    p_test.add_argument("--schema_file", type=str, required=True)
    p_test.add_argument("--output_dir", type=str, default="artifacts_window_typed")
    p_test.add_argument("--window_len", type=int, default=None)
    p_test.add_argument("--stride", type=int, default=None)
    p_test.add_argument("--feature_windows", type=str, default=None)
    p_test.add_argument("--allow_short_last", action="store_true")

    args = parser.parse_args()
    if args.mode == "train":
        build_train_dataset(
            train_root=Path(args.train_root),
            output_dir=Path(args.output_dir),
            window_len=args.window_len,
            stride=args.stride,
            feature_windows=parse_windows(args.feature_windows),
            allow_short_last=args.allow_short_last,
            type_config=args.type_config,
        )
    elif args.mode == "test":
        feature_windows = parse_windows(args.feature_windows) if args.feature_windows else None
        allow_short_last = True if args.allow_short_last else None
        build_test_dataset(
            test_root=Path(args.test_root),
            schema_file=Path(args.schema_file),
            output_dir=Path(args.output_dir),
            window_len=args.window_len,
            stride=args.stride,
            feature_windows=feature_windows,
            allow_short_last=allow_short_last,
        )


if __name__ == "__main__":
    main()
