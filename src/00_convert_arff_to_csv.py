import os
import pandas as pd
import arff  # liac-arff
from utils import ensure_dir, infer_label_column

DATA_ARFF = "data/twitter_spam.arff"
OUT_CSV = "data/twitter_spam.csv"


def main():
    if not os.path.exists(DATA_ARFF):
        raise FileNotFoundError(f"找不到 {DATA_ARFF}，請先把 ARFF 放到 data/ 目錄下。")

    with open(DATA_ARFF, "r", encoding="utf-8", errors="ignore") as f:
        obj = arff.load(f)

    attrs = [a[0] for a in obj["attributes"]]
    data = obj["data"]
    df = pd.DataFrame(data, columns=attrs)

    # 僅提示：最後一欄是標籤（spammer/non-spammer）
    label_col = infer_label_column(df)
    print("Columns:", df.columns.tolist())
    print("Inferred label column:", label_col)
    print(df[label_col].value_counts(dropna=False).head())

    ensure_dir(os.path.dirname(OUT_CSV))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()


