import sys

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 42
MAX_LEN = 64
N_BOOTSTRAP = 1000

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
y_true = (test["label"] == "spam").astype(int).to_numpy()
print("Kich thuoc test:", len(test), "| so spam:", int(y_true.sum()))

# SVM: hoc tren train, du doan tren test
svm = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
])
svm.fit(train["text"], train["label"])
pred_svm = (svm.predict(test["text"]) == "spam").astype(int)

# DistilBERT: nap ban da luu o buoc 3f
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("models/distilbert")
bert = AutoModelForSequenceClassification.from_pretrained("models/distilbert")
bert.to(device).eval()

preds = []
texts = test["text"].astype(str).tolist()
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        preds.append(bert(**enc).logits.argmax(dim=-1).cpu())
pred_bert = torch.cat(preds).numpy()


def danh_gia(ten, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label=1, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / len(y_true)

    # Bootstrap: lay mau lai co hoan lai de uoc luong khoang tin cay cua F1
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    f1_list = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        f1_list.append(f1_score(y_true[idx], pred[idx], zero_division=0))
    lo, hi = np.percentile(f1_list, [2.5, 97.5])

    print(f"\n=== {ten} (TAP TEST) ===")
    print(f"Ma tran nham lan: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Accuracy {acc:.3f} | Precision {p:.3f} | Recall {r:.3f} | "
          f"F1 {f1:.3f}")
    print(f"F1 khoang tin cay 95% (bootstrap): [{lo:.3f}; {hi:.3f}]")
    return {
        "model": ten,
        "accuracy": acc,
        "precision_spam": p,
        "recall_spam": r,
        "f1_spam": f1,
        "f1_ci95_thap": lo,
        "f1_ci95_cao": hi,
        "chan_nham_FP": int(fp),
        "bo_sot_FN": int(fn),
    }


rows = [
    danh_gia("DistilBERT (mo hinh chinh)", pred_bert),
    danh_gia("TF-IDF + SVM (mo hinh nhe)", pred_svm),
]
out = pd.DataFrame(rows).round(3)
out.to_csv("reports/ket_qua_test.csv", index=False)
print("\n", out.to_string(index=False))
print("\nLUU Y: khong chinh sua mo hinh sau khi xem ket qua test.")