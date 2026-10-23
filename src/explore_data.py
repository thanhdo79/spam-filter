import csv
import pandas as pd

df = pd.read_csv(
    "data/raw/SMSSpamCollection",
    sep="\t", header=None, names=["label", "text"],
    encoding="utf-8", encoding_errors="replace",
    quoting=csv.QUOTE_NONE,
)

print("So dong:", len(df))
print(df["label"].value_counts())
print("Ti le spam: {:.1%}".format((df["label"] == "spam").mean()))
print("Accuracy neu luon doan ham: {:.1%}".format((df["label"] == "ham").mean()))

df["length"] = df["text"].str.len()
print(df.groupby("label")["length"].describe())

print("So tin trung lap (cung noi dung, cung nhan):", df.duplicated().sum())
n_labels = df.groupby("text")["label"].nunique()
print("So noi dung vua la ham vua la spam:", (n_labels > 1).sum())

dedup = df.drop_duplicates()
print("Sau khi bo trung lap:", len(dedup))
print(dedup["label"].value_counts())
