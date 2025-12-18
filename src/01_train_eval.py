import os
import pandas as pd

from utils import (
    ensure_dir, save_json,
    FEATURE_GROUPS_DEFAULT,
    infer_label_column, basic_preprocess, split_data,
    select_columns, build_models, eval_binary,
    get_scores, best_threshold_for_precision, binarize_by_threshold, confusion_metrics,
    perm_importance
)

DATA_CSV = "data/twitter_spam.csv"
OUT_DIR = "outputs/01_train_eval"
SEED = 42


def train_one_setting(df_all, feature_cols, setting_name):
    label_col = infer_label_column(df_all)
    df = df_all[feature_cols + [label_col]].copy()

    # 預處理（數值化、缺失填補）
    X_df = basic_preprocess(df.drop(columns=[label_col]))
    df_prep = pd.concat([X_df, df[[label_col]]], axis=1)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df_prep, label_col, seed=SEED)

    models = build_models(random_state=SEED)
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        res = {
            "val": eval_binary(model, X_val, y_val),
            "test": eval_binary(model, X_test, y_test),
        }

        # 在驗證集上找 Precision>=0.95 的閾值，便於干預策略
        val_scores = get_scores(model, X_val)
        th = best_threshold_for_precision(y_val, val_scores, target_precision=0.95)
        res["selected_threshold_on_val"] = th

        # 在 test 上用該閾值計算混淆矩陣指標
        test_scores = get_scores(model, X_test)
        y_pred = binarize_by_threshold(test_scores, th["threshold"])
        res["test_confusion_at_selected_threshold"] = confusion_metrics(y_test, y_pred)

        # 保存模型
        model_path = os.path.join(OUT_DIR, f"model_{setting_name}_{model_name}.joblib")
        ensure_dir(OUT_DIR)
        import joblib
        joblib.dump({"model": model, "features": feature_cols}, model_path)
        res["model_path"] = model_path

        # permutation importance（可解釋性）
        try:
            imp = perm_importance(model, X_val, y_val, n_repeats=8, seed=SEED)
            imp_path = os.path.join(OUT_DIR, f"perm_importance_{setting_name}_{model_name}.csv")
            imp.to_csv(imp_path, index=False, encoding="utf-8")
            res["perm_importance_path"] = imp_path
        except Exception as e:
            res["perm_importance_error"] = str(e)

        results[model_name] = res

    return results


def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError("請先執行 src/00_convert_arff_to_csv.py 生成 data/twitter_spam.csv")

    df_all = pd.read_csv(DATA_CSV)
    label_col = infer_label_column(df_all)

    # 三個實驗設定：內容 / 社會 / 融合
    group_defs = FEATURE_GROUPS_DEFAULT

    content_cols, miss_c = select_columns(df_all, group_defs["content"])
    social_cols, miss_s = select_columns(df_all, group_defs["social_account"] + group_defs["social_interaction"])
    fusion_cols, miss_f = select_columns(df_all, group_defs["content"] + group_defs["social_account"] + group_defs["social_interaction"])

    meta = {
        "label_col": label_col,
        "content_cols": content_cols, "missing_content_cols": miss_c,
        "social_cols": social_cols, "missing_social_cols": miss_s,
        "fusion_cols": fusion_cols, "missing_fusion_cols": miss_f
    }
    ensure_dir(OUT_DIR)
    save_json(meta, os.path.join(OUT_DIR, "feature_meta.json"))

    all_results = {}
    all_results["content_only"] = train_one_setting(df_all, content_cols, "content_only")
    all_results["social_only"] = train_one_setting(df_all, social_cols, "social_only")
    all_results["fusion_all"] = train_one_setting(df_all, fusion_cols, "fusion_all")

    save_json(all_results, os.path.join(OUT_DIR, "metrics.json"))

    # 另存一份扁平表，便於寫報告
    rows = []
    for setting, mres in all_results.items():
        for model_name, res in mres.items():
            rows.append({
                "setting": setting,
                "model": model_name,
                "val_roc_auc": res["val"]["roc_auc"],
                "val_pr_auc": res["val"]["pr_auc"],
                "val_recall_at_p95": res["val"]["recall_at_precision_0.95"],
                "test_roc_auc": res["test"]["roc_auc"],
                "test_pr_auc": res["test"]["pr_auc"],
                "test_recall_at_p95": res["test"]["recall_at_precision_0.95"],
                "selected_threshold": res["selected_threshold_on_val"]["threshold"],
                "test_precision_at_selected_threshold": res["test_confusion_at_selected_threshold"]["precision"],
                "test_recall_at_selected_threshold": res["test_confusion_at_selected_threshold"]["recall"],
                "test_fpr_at_selected_threshold": res["test_confusion_at_selected_threshold"]["false_positive_rate"],
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "metrics_flat.csv"), index=False, encoding="utf-8")
    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()


