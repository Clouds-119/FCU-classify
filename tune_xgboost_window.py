# -*- coding: utf-8 -*-
"""
tune_xgboost_window.py

针对当前窗口级类型化特征的 XGBoost 小范围调参脚本。

前提：
1. 已生成窗口级类型化特征：
   artifacts_window_typed_v2/window_train_features.csv
   artifacts_window_typed_v2/window_schema.json

2. 当前目录下有修复后的训练脚本：
   window_train_only_fixed.py

本脚本会复用 window_train_only_fixed.py 里的数据读取、GroupKFold、文件级聚合等函数，
只单独调 XGBoost 参数。

推荐运行：
python tune_xgboost_window.py --data_dir "artifacts_window_typed_v2" --output_dir "artifacts_xgb_tune" --k_list 275,300,325 --n_iter 30 --cv 5

更省时间：
python tune_xgboost_window.py --data_dir "artifacts_window_typed_v2" --output_dir "artifacts_xgb_tune_k300" --k_list 300 --n_iter 30 --cv 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from window_train_only_fixed import (
    load_train_data,
    get_group_labels,
    make_group_cv_splits,
    align_proba,
    aggregate_to_file_level,
    build_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_int_list(s: str):
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))


def make_xgb(params: dict, random_state: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        **params,
    )


def baseline_params() -> dict:
    """当前原训练脚本里的 XGBoost 基线参数。"""
    return {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
    }


def build_param_candidates(n_iter: int, random_state: int):
    """
    小范围搜索，不做过大的暴力搜索，避免对 CV 过拟合。
    """
    grid = {
        "n_estimators": [300, 400, 500, 650],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.03, 0.04, 0.05],
        "subsample": [0.85, 0.9, 1.0],
        "colsample_bytree": [0.75, 0.8, 0.9],
        "reg_alpha": [0.0, 0.03, 0.05, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 2, 3],
        "gamma": [0.0, 0.03, 0.05, 0.1],
    }

    keys = list(grid.keys())
    all_params = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]

    rng = random.Random(random_state)
    sampled = rng.sample(all_params, k=min(n_iter, len(all_params)))

    # 基线参数必跑
    out = [baseline_params()]
    for p in sampled:
        if p not in out:
            out.append(p)
    return out


def evaluate_one_setting(params, k, X, y, groups, source_files, splits, global_classes, random_state):
    fold_rows = []
    fold_preds = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = make_xgb(params, random_state=random_state)
        pipe = build_pipeline(model, k=k, random_state=random_state)

        pipe.fit(X.iloc[train_idx], y[train_idx])

        proba = pipe.predict_proba(X.iloc[valid_idx])
        classes = list(pipe.classes_)
        proba = align_proba(proba, classes, global_classes)

        file_pred = aggregate_to_file_level(
            proba=proba,
            groups=groups[valid_idx],
            source_files=source_files[valid_idx],
            classes=global_classes,
            y_window=y[valid_idx],
        )

        y_true_file = file_pred["label"].to_numpy()
        y_pred_file = file_pred["predicted_label"].to_numpy()

        fold_rows.append({
            "fold": fold_id,
            "n_valid_files": int(len(file_pred)),
            "n_valid_windows": int(len(valid_idx)),
            "accuracy": float(accuracy_score(y_true_file, y_pred_file)),
            "macro_f1": float(f1_score(y_true_file, y_pred_file, average="macro")),
            "weighted_f1": float(f1_score(y_true_file, y_pred_file, average="weighted")),
        })

        file_pred["fold"] = fold_id
        file_pred["k"] = k
        for pk, pv in params.items():
            file_pred[f"param_{pk}"] = pv
        fold_preds.append(file_pred)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(fold_preds, ignore_index=True)

    res = {
        "k": int(k),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(fold_df["macro_f1"].mean()),
        "macro_f1_std": float(fold_df["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(fold_df["weighted_f1"].mean()),
        "weighted_f1_std": float(fold_df["weighted_f1"].std(ddof=0)),
    }
    res.update(params)
    return res, pred_df


def main():
    parser = argparse.ArgumentParser(description="窗口级类型化特征下的 XGBoost 小范围调参")
    parser.add_argument("--data_dir", type=str, default="artifacts_window_typed_v2")
    parser.add_argument("--output_dir", type=str, default="artifacts_xgb_tune")
    parser.add_argument("--k_list", type=str, default="275,300,325")
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, source_files, feature_names, schema = load_train_data(data_dir)
    global_classes = sorted(np.unique(y).tolist())

    print("\n========== XGBoost 小范围调参：数据检查 ==========")
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

    k_list = parse_int_list(args.k_list)
    param_list = build_param_candidates(args.n_iter, args.random_state)

    print("\n========== 搜索设置 ==========")
    print(f"CV 方法: {cv_name}, 折数: {n_splits}")
    print(f"K 候选: {k_list}")
    print(f"参数组合数: {len(param_list)}，其中第 1 组为基线参数")
    print(f"总实验数: {len(k_list) * len(param_list)}")

    results = []
    best_score = -1.0
    best_std = 999.0
    best_k = None
    best_params = None
    best_pred_df = None

    exp_id = 0
    total = len(k_list) * len(param_list)

    for k in k_list:
        for params in param_list:
            exp_id += 1
            print(f"\n[{exp_id}/{total}] XGBoost_k{k}, params={params}")

            try:
                res, pred_df = evaluate_one_setting(
                    params=params,
                    k=k,
                    X=X,
                    y=y,
                    groups=groups,
                    source_files=source_files,
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

                # 主目标：macro-F1；若均值几乎相同，则选 std 更小的
                if (res["macro_f1_mean"] > best_score + 1e-12) or (
                    abs(res["macro_f1_mean"] - best_score) <= 1e-12
                    and res["macro_f1_std"] < best_std
                ):
                    best_score = res["macro_f1_mean"]
                    best_std = res["macro_f1_std"]
                    best_k = k
                    best_params = params.copy()
                    best_pred_df = pred_df.copy()
                    print("  >>> 当前最优 XGBoost 参数已更新")

            except Exception as e:
                print(f"  实验失败，已跳过：{e}")
                fail = {"k": int(k), "error": str(e)}
                fail.update(params)
                results.append(fail)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["macro_f1_mean", "macro_f1_std"],
        ascending=[False, True],
        na_position="last",
    )

    results_path = output_dir / "xgb_tune_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    print("\n========== XGBoost 调参 Top 10 ==========")
    print(results_df.head(10).to_string(index=False))

    if best_params is None:
        raise RuntimeError("所有 XGBoost 参数实验均失败。")

    # 保存 OOF 文件级预测、分类报告、混淆矩阵
    oof_path = output_dir / "best_xgb_cv_file_predictions.csv"
    best_pred_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    y_true = best_pred_df["label"].to_numpy()
    y_pred = best_pred_df["predicted_label"].to_numpy()

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=global_classes,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report_path = output_dir / "best_xgb_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=global_classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in global_classes],
        columns=[f"pred_{c}" for c in global_classes],
    )
    cm_path = output_dir / "best_xgb_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print("\n========== 使用全部窗口样本训练最终 XGBoost ==========")
    print(f"最优 K: {best_k}")
    print(f"最优 macro-F1: {best_score:.4f} ± {best_std:.4f}")
    print(f"最优参数: {best_params}")

    final_model = make_xgb(best_params, random_state=args.random_state)
    final_pipe = build_pipeline(final_model, k=best_k, random_state=args.random_state)
    final_pipe.fit(X, y)

    model_path = output_dir / "best_xgb_window_pipeline.pkl"
    artifact = {
        "pipeline": final_pipe,
        "model_name": "XGBoost_tuned",
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "best_macro_f1_std": float(best_std),
        "best_params": best_params,
        "feature_names": feature_names,
        "class_labels": global_classes,
        "schema": schema,
        "aggregation": "mean_probability_by_group",
        "group_col": schema.get("group_col", "group_id"),
        "source_file_col": "source_file",
        "cv_results": results,
    }
    joblib.dump(artifact, model_path)

    info = {
        "model_name": "XGBoost_tuned",
        "k": int(best_k),
        "best_macro_f1": float(best_score),
        "best_macro_f1_std": float(best_std),
        "best_params": best_params,
        "n_window_samples": int(X.shape[0]),
        "n_files": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "cv_method": cv_name,
        "n_splits": int(n_splits),
        "outputs": {
            "model": str(model_path),
            "tune_results": str(results_path),
            "oof_file_predictions": str(oof_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(cm_path),
        },
    }

    info_path = output_dir / "xgb_tune_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n========== XGBoost 调参完成 ==========")
    print(f"模型文件: {model_path}")
    print(f"调参结果: {results_path}")
    print(f"OOF 文件级预测: {oof_path}")
    print(f"分类报告: {report_path}")
    print(f"混淆矩阵: {cm_path}")
    print(f"调参信息: {info_path}")


if __name__ == "__main__":
    main()
