import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib


FEATURE_GROUPS_DEFAULT = {
    # 來自 NSCLab 頁面列出的特徵名（你也可以按需要增減）
    "social_account": [
        "account_age", "no_follower", "no_following", "no_userfavourites", "no_lists", "no_tweets"
    ],
    "social_interaction": [
        "no_retweets", "no_usermention"
    ],
    "content": [
        "no_hashtag", "no_urls", "no_char", "no_digits"
    ]
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_label_column(df: pd.DataFrame) -> str:
    # NSCLab: 最後一列是類別
    return df.columns[-1]


def normalize_label(y: pd.Series) -> np.ndarray:
    """
    將原始字串標籤轉成 0/1：
    - 類似 'spammer' 的標記 -> 1
    - 類似 'non-spammer' 的標記 -> 0

    注意：不能直接用「是否包含 spammer」來判斷，
    因為 'non-spammer' 也包含 'spammer' 子字串，會全部被判成 1。
    """
    y_str = y.astype(str).str.strip().str.lower()

    y_bin = np.zeros(len(y_str), dtype=int)

    # 先標記明確的非垃圾帳號
    mask_non = y_str.isin(["non-spammer", "non_spammer", "legit", "ham"])
    y_bin[mask_non] = 0

    # 再標記明確的垃圾帳號
    mask_spam = y_str.isin(["spammer", "spam"])
    y_bin[mask_spam] = 1

    # 其他未識別情況可以依需求調整，暫時保持為 0
    return y_bin


def split_data(df: pd.DataFrame, label_col: str, test_size=0.15, val_size=0.15, seed=42):
    X = df.drop(columns=[label_col]).copy()
    y = normalize_label(df[label_col])

    # 先切出 test，再從 train_val 切 val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # val 佔總量 val_size，則在 trainval 內比例為 val_size/(1-test_size)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=seed, stratify=y_trainval
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 數值欄：缺失用中位數填補；同時把所有欄轉為數值（ARFF 讀入可能是 object）
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in out.columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    return out


def select_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    # 回傳存在於 df 中的欄位（避免欄位名不匹配直接報錯）
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    return existing, missing


def build_models(random_state=42):
    # M1: Logistic Regression（可解釋基線）
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ))
    ])

    # M2: HistGradientBoosting（樹模型，強非線性，sklearn 自帶，部署簡單）
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_iter=300,
        random_state=random_state
    )

    return {
        "LR": lr,
        "HGB": hgb
    }


def eval_binary(model, X, y_true) -> dict:
    # 相容 sklearn 的 predict_proba 或 decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)

    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # 計算 Recall@Precision>=0.95（找滿足精度閾值的最大召回）
    mask = prec >= 0.95
    recall_at_p95 = float(np.max(rec[mask])) if np.any(mask) else 0.0

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "recall_at_precision_0.95": recall_at_p95,
        "n": int(len(y_true))
    }


def best_threshold_for_precision(y_true, y_score, target_precision=0.95):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve 的 thr 長度 = len(prec) - 1
    # 找 precision >= target 的點，對應閾值取最大召回
    best = None
    for i in range(len(thr)):
        p = prec[i]
        r = rec[i]
        t = thr[i]
        if p >= target_precision:
            if (best is None) or (r > best["recall"]):
                best = {"threshold": float(t), "precision": float(p), "recall": float(r)}
    if best is None:
        # 找不到滿足 precision 的閾值，就回傳使 precision 最大的閾值點
        i = int(np.argmax(prec[:-1]))
        best = {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i])}
    return best


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def binarize_by_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def confusion_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr)
    }


def perm_importance(model, X_val, y_val, n_repeats=10, seed=42):
    r = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="average_precision"
    )
    imp = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)
    return imp


