# -*- coding: utf-8 -*-
"""
window_data_preprocess.py

窗口级 FCU 数据处理脚本。

功能：
1. 将每个 2 天运行文件切分为多个连续时间窗口；
2. 每个窗口继承原始文件标签；
3. 对每个窗口复用原 data_preprocess_accuracy_v2.py 中的多特征融合函数 extract_features；
4. 保存窗口级训练特征、测试特征和 schema；
5. 保留 group_id=原始文件名，供后续 GroupKFold 使用，避免数据泄漏。

注意：
后续训练时不能普通随机划分窗口样本，必须按 group_id 分组交叉验证。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from data_preprocess_accuracy_v2 import read_table, extract_features

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


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


def infer_feature_columns_from_train(train_root: Path) -> List[str]:
    """
    从训练集所有文件中推断可转为数值的原始变量列。
    时间列、索引列会被排除。
    """
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


def make_window_slices(
    n_rows: int,
    window_len: int,
    stride: int,
    allow_short_last: bool = False,
    min_short_ratio: float = 0.8,
) -> List[Tuple[int, int]]:
    """
    返回窗口起止索引列表，end 不包含。
    """
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


def extract_window_features_for_file(
    file_path: Path,
    feature_columns: List[str],
    feature_windows: Tuple[int, ...],
    window_len: int,
    stride: int,
    label: Optional[int] = None,
    allow_short_last: bool = False,
) -> List[Dict]:
    df = read_table(file_path)
    df = sort_by_time_if_possible(df)

    slices = make_window_slices(
        n_rows=len(df),
        window_len=window_len,
        stride=stride,
        allow_short_last=allow_short_last,
    )

    rows = []

    for win_id, (start, end) in enumerate(slices):
        wdf = df.iloc[start:end].reset_index(drop=True)

        feats = extract_features(
            wdf,
            feature_columns=feature_columns,
            window_sizes=feature_windows,
        )

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


def build_train_window_dataset(
    train_root: Path,
    output_dir: Path,
    window_len: int,
    stride: int,
    feature_windows: Tuple[int, ...],
    allow_short_last: bool = False,
) -> None:
    if not train_root.exists():
        raise FileNotFoundError(f"训练集路径不存在：{train_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not label_dirs:
        raise RuntimeError(f"训练集目录下没有类别子文件夹：{train_root}")

    print("\n========== 推断训练集原始变量列 ==========")
    feature_columns = infer_feature_columns_from_train(train_root)
    print(f"推断得到原始变量列数: {len(feature_columns)}")
    print(f"前 10 个变量列: {feature_columns[:10]}")

    all_rows = []
    file_summary_rows = []

    print("\n========== 开始生成训练集窗口特征 ==========")

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
                rows = extract_window_features_for_file(
                    file_path=fp,
                    feature_columns=feature_columns,
                    feature_windows=feature_windows,
                    window_len=window_len,
                    stride=stride,
                    label=label,
                    allow_short_last=allow_short_last,
                )
            except Exception as e:
                print(f"警告：处理失败，跳过 {fp}: {e}")
                continue

            all_rows.extend(rows)
            file_summary_rows.append({
                "source_file": fp.name,
                "group_id": fp.stem,
                "label": label,
                "n_windows": len(rows),
            })

    if not all_rows:
        raise RuntimeError("没有生成任何窗口样本，请检查窗口长度、步长和数据文件。")

    df_win = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan)
    df_summary = pd.DataFrame(file_summary_rows)

    meta_cols = ["source_file", "group_id", "window_id", "start_idx", "end_idx", "window_n_rows", "label"]
    feature_names = [c for c in df_win.columns if c not in meta_cols]

    all_nan_features = [c for c in feature_names if df_win[c].isna().all()]
    if all_nan_features:
        print(f"删除全 NaN 窗口特征 {len(all_nan_features)} 个，例如：{all_nan_features[:10]}")
        df_win = df_win.drop(columns=all_nan_features)
        feature_names = [c for c in feature_names if c not in all_nan_features]

    train_csv = output_dir / "window_train_features.csv"
    summary_csv = output_dir / "window_train_file_summary.csv"
    schema_json = output_dir / "window_schema.json"

    df_win.to_csv(train_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    schema = {
        "task": "window_level_fcu_preprocess",
        "train_root": str(train_root),
        "window_len": int(window_len),
        "stride": int(stride),
        "feature_windows": list(feature_windows),
        "allow_short_last": bool(allow_short_last),
        "feature_columns": feature_columns,
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

    print("\n========== 训练窗口数据生成完成 ==========")
    print(f"文件级样本数: {df_summary.shape[0]}")
    print(f"窗口级样本数: {df_win.shape[0]}")
    print(f"窗口特征维度: {len(feature_names)}")
    print("文件级类别分布:", dict(sorted(Counter(df_summary["label"]).items())))
    print("窗口级类别分布:", dict(sorted(Counter(df_win["label"]).items())))
    print(f"训练窗口特征表: {train_csv}")
    print(f"文件窗口统计表: {summary_csv}")
    print(f"schema 文件: {schema_json}")


def build_test_window_dataset(
    test_root: Path,
    schema_file: Path,
    output_dir: Path,
    window_len: Optional[int] = None,
    stride: Optional[int] = None,
    feature_windows: Optional[Tuple[int, ...]] = None,
    allow_short_last: Optional[bool] = None,
) -> None:
    if not test_root.exists():
        raise FileNotFoundError(f"测试集路径不存在：{test_root}")
    if not schema_file.exists():
        raise FileNotFoundError(f"schema 文件不存在：{schema_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_columns = schema["feature_columns"]
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

    print("\n========== 开始生成测试集窗口特征 ==========")
    print(f"测试文件数量: {len(files)}")
    print(f"窗口长度: {window_len}, 步长: {stride}, 特征滚动窗口: {feature_windows}")

    for fp in files:
        try:
            rows = extract_window_features_for_file(
                file_path=fp,
                feature_columns=feature_columns,
                feature_windows=feature_windows,
                window_len=window_len,
                stride=stride,
                label=None,
                allow_short_last=allow_short_last,
            )
        except Exception as e:
            print(f"警告：处理失败，跳过 {fp}: {e}")
            continue

        all_rows.extend(rows)
        file_summary_rows.append({
            "source_file": fp.name,
            "group_id": fp.stem,
            "n_windows": len(rows),
        })

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

    print("\n========== 测试窗口数据生成完成 ==========")
    print(f"测试文件数: {df_summary.shape[0]}")
    print(f"测试窗口样本数: {df_aligned.shape[0]}")
    print(f"测试窗口特征维度: {len(train_feature_names)}")
    print(f"测试窗口特征表: {test_csv}")
    print(f"测试文件窗口统计表: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="窗口级 FCU 数据处理脚本")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_train = subparsers.add_parser("train", help="生成训练集窗口特征")
    p_train.add_argument("--train_root", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="artifacts_window_data")
    p_train.add_argument("--window_len", type=int, default=240)
    p_train.add_argument("--stride", type=int, default=60)
    p_train.add_argument("--feature_windows", type=str, default="5,15,30")
    p_train.add_argument("--allow_short_last", action="store_true")

    p_test = subparsers.add_parser("test", help="生成测试集窗口特征")
    p_test.add_argument("--test_root", type=str, required=True)
    p_test.add_argument("--schema_file", type=str, required=True)
    p_test.add_argument("--output_dir", type=str, default="artifacts_window_data")
    p_test.add_argument("--window_len", type=int, default=None)
    p_test.add_argument("--stride", type=int, default=None)
    p_test.add_argument("--feature_windows", type=str, default=None)
    p_test.add_argument("--allow_short_last", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        build_train_window_dataset(
            train_root=Path(args.train_root),
            output_dir=Path(args.output_dir),
            window_len=args.window_len,
            stride=args.stride,
            feature_windows=parse_windows(args.feature_windows),
            allow_short_last=args.allow_short_last,
        )

    elif args.mode == "test":
        feature_windows = parse_windows(args.feature_windows) if args.feature_windows else None
        allow_short_last = True if args.allow_short_last else None

        build_test_window_dataset(
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
