import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "data/raw/SMSSpamCollection",
    sep="\t", header=None, names=["label", "text"],
    encoding="utf-8", encoding_errors="replace",
    quoting=csv.QUOTE_NONE,
)
print("Ban dau:", len(df))
df = df.drop_duplicates().reset_index(drop=True)
print("Sau loai trung:", len(df))

# 70% train, 30% con lai chia doi thanh val va test
train, temp = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.to_csv(OUT_DIR / f"{name}.csv", index=False)
    n_spam = int((part["label"] == "spam").sum())
    print(f"{name}: {len(part)} tin, spam {n_spam} ({n_spam / len(part):.1%})")

# Kiem tra ro ri: khong duoc co tin nao nam o hai tap
s_train, s_val, s_test = set(train["text"]), set(val["text"]), set(test["text"])
print("Trung train-val:", len(s_train & s_val))
print("Trung train-test:", len(s_train & s_test))
print("Trung val-test:", len(s_val & s_test))