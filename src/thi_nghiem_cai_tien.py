import sys

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200)

SEED = 42
MAX_LEN = 64
THU_MUC = "models/distilbert"

DONG_KET_QUA = []


def in_ra(*args):
    """In ra man hinh va luu lai de ghi ra tep bao cao."""
    dong = " ".join(str(a) for a in args)
    print(dong)
    DONG_KET_QUA.append(dong)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(THU_MUC)
bert = AutoModelForSequenceClassification.from_pretrained(THU_MUC)
bert.to(device).eval()


def xac_suat_bert(texts, max_len):
    ket_qua = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=max_len,
                      padding="max_length", return_tensors="pt").to(device)
            logits = bert(**enc).logits
            ket_qua.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(ket_qua).numpy()


train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")
for df in (train, val, test):
    df["text"] = df["text"].astype(str)
    df["y"] = (df["label"] == "spam").astype(int)

val["p_bert"] = xac_suat_bert(val["text"].tolist(), MAX_LEN)
test["p_bert"] = xac_suat_bert(test["text"].tolist(), MAX_LEN)

vt = pd.concat([val.assign(split="val"), test.assign(split="test")],
               ignore_index=True)
vt["pred"] = (vt["p_bert"] >= 0.5).astype(int)
vt["loi"] = vt["pred"] != vt["y"]

# ---------- A. Viec cat 64 token ----------
in_ra("=== A. VIEC CAT TOI DA 64 TOKEN (val + test, chi mo ta) ===")
vt["so_token"] = [len(tok(t, truncation=False)["input_ids"]) for t in vt["text"]]
in_ra(f"So token lon nhat: {vt['so_token'].max()} | trung vi: {vt['so_token'].median()}")
dai = vt[vt["so_token"] > MAX_LEN]
in_ra(f"Tin vuot {MAX_LEN} token: {len(dai)}/{len(vt)} ({len(dai) / len(vt) * 100:.1f}%)")
if len(dai) > 0:
    in_ra(f"Trong do: {int(dai['y'].sum())} tin spam, {int(dai['loi'].sum())} tin bi sai")
    p128 = xac_suat_bert(dai["text"].tolist(), 128)
    pred128 = (p128 >= 0.5).astype(int)
    doi = int((pred128 != dai["pred"].to_numpy()).sum())
    loi128 = int((pred128 != dai["y"].to_numpy()).sum())
    in_ra(f"Cho doc toi da 128 token (KHONG huan luyen lai): {doi} tin doi du doan, "
          f"so loi {int(dai['loi'].sum())} -> {loi128}")

# ---------- B. Do tin cay cua xac suat ----------
in_ra("\n=== B. XAC SUAT SPAM CO DANG TIN KHONG (val + test, chi mo ta) ===")
moc = [0, 0.1, 0.3, 0.7, 0.9, 1.0001]
ten_khoang = ["[0-0.1)", "[0.1-0.3)", "[0.3-0.7)", "[0.7-0.9)", "[0.9-1]"]
vt["khoang"] = pd.cut(vt["p_bert"], bins=moc, labels=ten_khoang, right=False)
bang_b = vt.groupby("khoang", observed=True).agg(
    so_tin=("y", "size"), so_spam_that=("y", "sum"))
bang_b["ty_le_spam_that_%"] = (bang_b["so_spam_that"] / bang_b["so_tin"] * 100).round(1)
in_ra(bang_b.to_string())

# ---------- C. Do phu spam noi dung nguoi lon ----------
in_ra("\n=== C. SPAM NOI DUNG NGUOI LON (tu khoa tho, chi mang tinh goi y) ===")
tu_khoa = ["xxx", "babe", "sexy", "horny", "nude", "porn", "naughty", "nasty", "filthy"]
mau = "|".join(tu_khoa)


def co_nguoi_lon(chuoi):
    return chuoi.str.lower().str.contains(mau)


tr_spam = train[train["y"] == 1]
tr_ham = train[train["y"] == 0]
in_ra(f"Train: {int(co_nguoi_lon(tr_spam['text']).sum())}/{len(tr_spam)} tin spam "
      f"va {int(co_nguoi_lon(tr_ham['text']).sum())}/{len(tr_ham)} tin ham co tu khoa")
spam_vt = vt[vt["y"] == 1]
nl = spam_vt[co_nguoi_lon(spam_vt["text"])]
in_ra(f"Val + test: {len(nl)} tin spam co tu khoa, bo sot {int((nl['pred'] == 0).sum())} "
      f"(recall {(nl['pred'] == 1).mean() * 100:.1f}%)")
in_ra(f"Recall chung cua spam val + test: {(spam_vt['pred'] == 1).mean() * 100:.1f}% "
      f"tren {len(spam_vt)} tin")

# ---------- D. Ket hop SVM voi DistilBERT (chi dung val) ----------
in_ra("\n=== D. KET HOP SVM + DISTILBERT (CHI DUNG TAP VAL) ===")
svm = CalibratedClassifierCV(
    Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
    ]),
    cv=5,
)
svm.fit(train["text"], train["label"])
cot_spam = list(svm.classes_).index("spam")
val["p_svm"] = svm.predict_proba(val["text"])[:, cot_spam]

b = val["p_bert"] >= 0.5
s = val["p_svm"] >= 0.5
cau_hinh = {
    "DistilBERT": b,
    "SVM (da hieu chinh xac suat)": s,
    "Trung binh hai xac suat": (val["p_bert"] + val["p_svm"]) / 2 >= 0.5,
    "OR (mot trong hai noi spam)": b | s,
    "AND (ca hai cung noi spam)": b & s,
}


def danh_gia(ten, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        val["y"], pred, pos_label=1, average="binary", zero_division=0)
    return {
        "cau_hinh": ten, "precision": round(p, 3), "recall": round(r, 3),
        "f1": round(f1, 3),
        "FP": int(((pred == 1) & (val["y"] == 0)).sum()),
        "FN": int(((pred == 0) & (val["y"] == 1)).sum()),
    }


bang_d = pd.DataFrame([danh_gia(t, p.astype(int)) for t, p in cau_hinh.items()])
in_ra(bang_d.to_string(index=False))
in_ra("\nLUU Y: day la thi nghiem tham do tren val; khong dung test de chon.")

with open("reports/thi_nghiem_cai_tien.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(DONG_KET_QUA) + "\n")
print("\nDa luu reports/thi_nghiem_cai_tien.txt")