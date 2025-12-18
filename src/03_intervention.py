import os
import numpy as np
import pandas as pd

from utils import (
    ensure_dir, infer_label_column, basic_preprocess, split_data,
    FEATURE_GROUPS_DEFAULT, select_columns,
    build_models, get_scores, binarize_by_threshold, confusion_metrics
)

DATA_CSV = "data/twitter_spam.csv"
OUT_DIR = "outputs/03_intervention"
SEED = 42


def main():
    ensure_dir(OUT_DIR)
    df_all = pd.read_csv(DATA_CSV)
    label_col = infer_label_column(df_all)

    # 用「融合全特徵」的模型做干預評估
    groups = FEATURE_GROUPS_DEFAULT
    fusion_cols, _ = select_columns(df_all, groups["content"] + groups["social_account"] + groups["social_interaction"])

    # 預處理
    df = df_all[fusion_cols + [label_col]].copy()
    X_df = basic_preprocess(df.drop(columns=[label_col]))
    df_prep = pd.concat([X_df, df[[label_col]]], axis=1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_prep, label_col, seed=SEED)

    # 訓練一個樹模型（更適合作為「風險分數」）
    model = build_models(random_state=SEED)["HGB"]
    model.fit(X_train, y_train)

    scores_test = get_scores(model, X_test)

    # 「預計傳播降低」：用 no_retweets 作為傳播強度代理（若欄位不存在則用 1 作為占位）
    if "no_retweets" in X_test.columns:
        spread_proxy = X_test["no_retweets"].to_numpy()
    else:
        spread_proxy = np.ones(len(X_test), dtype=float)

    # 閾值掃描：介於分數 1%~99% 分位數之間
    thresholds = np.linspace(np.quantile(scores_test, 0.01), np.quantile(scores_test, 0.99), 40)

    rows = []
    for t in thresholds:
        y_pred = binarize_by_threshold(scores_test, float(t))
        cm = confusion_metrics(y_test, y_pred)

        # 干預：對判為高風險 (1) 的樣本進行「降權／外鏈摩擦」
        intervene_mask = (y_pred == 1)

        # 預計攔截的 spam 比例（= recall）
        block_rate_spam = cm["recall"]
        # 誤傷比例（= FPR，表示正常被干預的比例）
        block_rate_normal = cm["false_positive_rate"]

        # 預計傳播降低：被干預樣本的傳播代理和 / 總傳播代理和
        total_spread = float(spread_proxy.sum()) if spread_proxy.sum() > 0 else 1.0
        reduced_spread_ratio = float(spread_proxy[intervene_mask].sum() / total_spread)

        rows.append({
            "threshold": float(t),
            "precision": cm["precision"],
            "recall": cm["recall"],
            "false_positive_rate": cm["false_positive_rate"],
            "intervened_ratio_all": float(intervene_mask.mean()),
            "estimated_spread_reduction_ratio": reduced_spread_ratio
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "intervention_threshold_scan.csv"), index=False, encoding="utf-8")

    # 選一個「precision>=0.95」下 spread reduction 最大的閾值（可依業務需求調整）
    cand = out[out["precision"] >= 0.95].copy()
    if len(cand) > 0:
        best = cand.sort_values("estimated_spread_reduction_ratio", ascending=False).head(1)
    else:
        best = out.sort_values("precision", ascending=False).head(1)
    best.to_csv(os.path.join(OUT_DIR, "intervention_best_rule.csv"), index=False, encoding="utf-8")

    print("Saved:", os.path.join(OUT_DIR, "intervention_threshold_scan.csv"))
    print("Saved:", os.path.join(OUT_DIR, "intervention_best_rule.csv"))


if __name__ == "__main__":
    main()


