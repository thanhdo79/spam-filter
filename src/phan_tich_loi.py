import sys

import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# In tieng Anh co ky tu la van hien thi duoc tren Terminal
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
MAX_LEN = 64

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
y_true = (val["label"] == "spam").astype(int).to_numpy()

# Mo hinh 1: SVM (hoc lai, rat nhanh, cung cau hinh nhu compare_models.py)
svm = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
])
svm.fit(train["text"], train["label"])
pred_svm = (svm.predict(val["text"]) == "spam").astype(int)

# Mo hinh 2: DistilBERT (nap lai ban tot nhat da luu o buoc 3f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("models/distilbert")
bert = AutoModelForSequenceClassification.from_pretrained("models/distilbert")
bert.to(device).eval()

preds = []
texts = val["text"].astype(str).tolist()
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        preds.append(bert(**enc).logits.argmax(dim=-1).cpu())
pred_bert = torch.cat(preds).numpy()

print("So loi SVM       :", int((pred_svm != y_true).sum()))
print("So loi DistilBERT:", int((pred_bert != y_true).sum()))

# Bang doi chieu theo tung tin
ok_svm = pred_svm == y_true
ok_bert = pred_bert == y_true
ca_hai_dung = int((ok_svm & ok_bert).sum())
ca_hai_sai = int((~ok_svm & ~ok_bert).sum())
b = int((ok_svm & ~ok_bert).sum())   # SVM dung, DistilBERT sai
c = int((~ok_svm & ok_bert).sum())   # SVM sai, DistilBERT dung
print(f"\nCa hai dung: {ca_hai_dung} | Ca hai sai: {ca_hai_sai}")
print(f"Chi SVM dung (b): {b} | Chi DistilBERT dung (c): {c}")

if b + c > 0:
    p_value = binomtest(min(b, c), b + c, 0.5).pvalue
    print(f"McNemar (chinh xac) p-value = {p_value:.4f}")
    print("=> Co y nghia thong ke (p < 0.05)" if p_value < 0.05
          else "=> CHUA co y nghia thong ke (p >= 0.05)")
else:
    print("Hai mo hinh sai o dung cac tin giong nhau, khong the kiem dinh")

# Liet ke cac tin bi doan sai boi it nhat mot mo hinh
loi = val.copy()
loi["svm"] = ["spam" if x else "ham" for x in pred_svm]
loi["distilbert"] = ["spam" if x else "ham" for x in pred_bert]
loi = loi[(pred_svm != y_true) | (pred_bert != y_true)]
loi = loi.rename(columns={"label": "nhan_that"})
loi.to_csv("reports/loi_val.csv", index=False)

print(f"\nCAC TIN BI SAI (tong {len(loi)} tin):")
for _, row in loi.iterrows():
    print(f"- that={row['nhan_that']} | svm={row['svm']} | "
          f"bert={row['distilbert']} | {str(row['text'])[:150]}")