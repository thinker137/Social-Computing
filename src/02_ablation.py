import os
import pandas as pd

from utils import (
    ensure_dir,
    FEATURE_GROUPS_DEFAULT,
    infer_label_column, basic_preprocess, split_data,
    select_columns, build_models, eval_binary
)

DATA_CSV = "data/twitter_spam.csv"
OUT_DIR = "outputs/02_ablation"
SEED = 42


def main():
    ensure_dir(OUT_DIR)
    df_all = pd.read_csv(DATA_CSV)
    label_col = infer_label_column(df_all)

    groups = FEATURE_GROUPS_DEFAULT
    content_cols, _ = select_columns(df_all, groups["content"])
    acct_cols, _ = select_columns(df_all, groups["social_account"])
    inter_cols, _ = select_columns(df_all, groups["social_interaction"])

    full_cols = content_cols + acct_cols + inter_cols

    ablations = {
        "full": full_cols,
        "minus_content": [c for c in full_cols if c not in content_cols],
        "minus_social_account": [c for c in full_cols if c not in acct_cols],
        "minus_social_interaction": [c for c in full_cols if c not in inter_cols],
    }

    rows = []
    for name, cols in ablations.items():
        df = df_all[cols + [label_col]].copy()
        X_df = basic_preprocess(df.drop(columns=[label_col]))
        df_prep = pd.concat([X_df, df[[label_col]]], axis=1)

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_prep, label_col, seed=SEED)

        models = build_models(random_state=SEED)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            val = eval_binary(model, X_val, y_val)
            test = eval_binary(model, X_test, y_test)
            rows.append({
                "ablation": name,
                "model": model_name,
                "n_features": len(cols),
                "val_roc_auc": val["roc_auc"],
                "val_pr_auc": val["pr_auc"],
                "val_recall_at_p95": val["recall_at_precision_0.95"],
                "test_roc_auc": test["roc_auc"],
                "test_pr_auc": test["pr_auc"],
                "test_recall_at_p95": test["recall_at_precision_0.95"],
            })

    out = pd.DataFrame(rows).sort_values(["model", "ablation"])
    out.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False, encoding="utf-8")
    print("Saved:", os.path.join(OUT_DIR, "ablation_results.csv"))


if __name__ == "__main__":
    main()


