import os
from typing import Dict, List

import arff  # liac-arff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ensure_dir, infer_label_column, normalize_label
from sklearn.metrics import roc_auc_score


BASE_OUT_DIR = "outputs/04_propagation"

# 兩個核心數據集：主實驗（B）與魯棒性復驗（C）
DATASETS: Dict[str, str] = {
    "B_95k_continuous": "data/ICC/95k-continuous.arff",
    "C_95k_random": "data/ICC/95k-random.arff",
}

# 作為「傳播／引流／社交牽引」代理的三個特徵
PROXY_FEATURES: List[str] = [
    "no_retweets",     # 傳播強度代理
    "no_urls",         # 引流強度代理
    "no_usermention",  # 社交牽引代理
]


def compute_sparsity(df: pd.DataFrame, feature: str, label_bin: np.ndarray) -> Dict[str, float]:
    """計算 P(x>0 | spam) 與 P(x>0 | non-spam)，用來顯示稀疏性。"""
    x = pd.to_numeric(df[feature], errors="coerce")
    mask_valid = x.notna()
    x = x[mask_valid]
    y = label_bin[mask_valid.to_numpy()]

    spam = x[y == 1]
    non_spam = x[y == 0]

    p_spam = float((spam > 0).mean()) if len(spam) > 0 else float("nan")
    p_non = float((non_spam > 0).mean()) if len(non_spam) > 0 else float("nan")

    return {
        "p_gt0_spam": p_spam,
        "p_gt0_nonspam": p_non,
    }


def plot_sparsity_bar(dataset_name: str, feature: str, sparsity: Dict[str, float], out_path: str):
    """Draw bar chart of P(x>0 | spam) vs P(x>0 | non-spam)."""
    labels = ["non-spammer", "spammer"]
    values = [sparsity["p_gt0_nonspam"], sparsity["p_gt0_spam"]]

    # 調大圖像尺寸與字體，並縮短標題，避免文字重疊
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4C72B0", "#DD8452"])
    plt.ylim(0, 1)
    plt.ylabel(f"P({feature} > 0)")
    # 使用簡短標題，減少與圖例／邊界重疊風險
    plt.title(f"{dataset_name}: {feature} > 0", fontsize=11)
    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_arff_to_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 ARFF 檔案：{path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        obj = arff.load(f)
    attrs = [a[0] for a in obj["attributes"]]
    data = obj["data"]
    df = pd.DataFrame(data, columns=attrs)
    return df


def plot_feature_distribution(df: pd.DataFrame, feature: str, label_bin: np.ndarray, out_path: str):
    # 只保留非負數值，避免極端或缺失影響圖形
    x = pd.to_numeric(df[feature], errors="coerce")
    mask_valid = x.notna() & (x >= 0)
    x = x[mask_valid]
    y = label_bin[mask_valid.to_numpy()]

    spam = x[y == 1]
    non_spam = x[y == 0]

    # 調大圖像尺寸，讓標題與圖例有足夠空間顯示
    plt.figure(figsize=(8, 5))
    # 使用對數刻度 bins，以便同時看低值與高值
    bins = np.logspace(
        np.log10(max(1e-3, x[x > 0].min() if (x > 0).any() else 1e-3)),
        np.log10(max(1.0, x.max())),
        40,
    )
    plt.hist(non_spam, bins=bins, alpha=0.6, label="non-spammer", density=True, color="#4C72B0")
    plt.hist(spam, bins=bins, alpha=0.6, label="spammer", density=True, color="#DD8452")
    plt.xscale("log")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"{feature} distribution (log-x)")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_feature(df: pd.DataFrame, feature: str, label_bin: np.ndarray) -> Dict[str, float]:
    x = pd.to_numeric(df[feature], errors="coerce")
    mask_valid = x.notna()
    x = x[mask_valid]
    y = label_bin[mask_valid.to_numpy()]

    spam = x[y == 1]
    non_spam = x[y == 0]

    def stats(arr: pd.Series, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_median": float(arr.median()),
            f"{prefix}_p25": float(arr.quantile(0.25)),
            f"{prefix}_p75": float(arr.quantile(0.75)),
        }

    out: Dict[str, float] = {}
    out.update(stats(non_spam, "nonspam"))
    out.update(stats(spam, "spam"))

    # spam / non-spam 的平均比值（>1 表示 spam 在該特徵上更「極端」）
    if out["nonspam_mean"] > 0:
        out["mean_ratio_spam_over_nonspam"] = float(out["spam_mean"] / out["nonspam_mean"])
    else:
        out["mean_ratio_spam_over_nonspam"] = np.nan

    # 單特徵作為風險分數時的 ROC-AUC（衡量該特徵區分 spam / non-spam 的能力）
    try:
        out["roc_auc_single_feature"] = float(roc_auc_score(y, x))
    except Exception:
        out["roc_auc_single_feature"] = np.nan

    return out


def analyze_one_dataset(name: str, path: str):
    print(f"Processing dataset: {name} ({path})")
    df = load_arff_to_df(path)
    label_col = infer_label_column(df)
    y_bin = normalize_label(df[label_col])

    # 基本資訊
    n = len(df)
    spam_ratio = float(y_bin.mean())
    print(f"  Samples: {n}, spam ratio: {spam_ratio:.4f}")

    rows = []
    out_dir = os.path.join(BASE_OUT_DIR, name)
    ensure_dir(out_dir)

    sparsity_rows = []

    for feat in PROXY_FEATURES:
        if feat not in df.columns:
            print(f"  [WARN] feature {feat} not in columns, skip.")
            continue

        # 圖：spam vs non-spam 分布
        fig_path = os.path.join(out_dir, f"{name}_dist_{feat}.png")
        plot_feature_distribution(df, feat, y_bin, fig_path)

        # 統計摘要
        summ = summarize_feature(df, feat, y_bin)
        row = {"dataset": name, "feature": feat, "n": n, "spam_ratio": spam_ratio}
        row.update(summ)
        rows.append(row)

        # 稀疏性統計 + 柱狀圖
        sp = compute_sparsity(df, feat, y_bin)
        sp_row = {"dataset": name, "feature": feat}
        sp_row.update(sp)
        sparsity_rows.append(sp_row)

        sp_fig_path = os.path.join(out_dir, f"{name}_sparsity_{feat}.png")
        plot_sparsity_bar(name, feat, sp, sp_fig_path)

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_path = os.path.join(out_dir, f"{name}_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        print("  Saved summary:", summary_path)

        if sparsity_rows:
            sparsity_df = pd.DataFrame(sparsity_rows)
            sparsity_path = os.path.join(out_dir, f"{name}_sparsity.csv")
            sparsity_df.to_csv(sparsity_path, index=False, encoding="utf-8")
            print("  Saved sparsity stats:", sparsity_path)

        # 生成一份簡要結論草稿（中文），方便直接寫進報告
        md_path = os.path.join(out_dir, f"{name}_conclusion_zh.md")
        lines = [
            f"# 數據集 {name} 的傳播／引流代理特徵分析結論草稿",
            "",
            f"- 總樣本數：{n}，spam 佔比約 {spam_ratio:.2%}。",
        ]
        for feat in PROXY_FEATURES:
            sub = summary_df[summary_df["feature"] == feat]
            if sub.empty:
                continue
            r = sub.iloc[0].copy()
            lines.append(
                f"- 特徵 `{feat}`：spam 的平均值約為 non-spam 的 "
                f"{r['mean_ratio_spam_over_nonspam']:.2f} 倍，"
                f"單特徵 ROC-AUC 約為 {r['roc_auc_single_feature']:.3f}。"
            )
        lines.append("")
        lines.append(
            "說明：上述三個特徵分別可視為「傳播強度（no_retweets）」「引流強度（no_urls）」與"
            "「社交牽引（no_usermention）」的統計代理；本分析基於統計分布與區分能力，"
            "屬於**統計層面的傳播模式**，"
        )
        lines.append("而非基於具體轉發鏈路或級聯樹的網路結構分析。")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("  Saved conclusion draft:", md_path)


def main():
    ensure_dir(BASE_OUT_DIR)
    for name, path in DATASETS.items():
        analyze_one_dataset(name, path)


if __name__ == "__main__":
    main()


