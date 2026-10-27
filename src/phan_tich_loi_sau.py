import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200)

MAX_LEN = 64
THU_MUC = "models/distilbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(THU_MUC)
model = AutoModelForSequenceClassification.from_pretrained(THU_MUC)
model.to(device).eval()


def xac_suat_spam(texts):
    ket_qua = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, max_length=MAX_LEN,
                      padding="max_length", return_tensors="pt").to(device)
            logits = model(**enc).logits
            ket_qua.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(ket_qua).numpy()


def chuan_bi(ten_tap):
    df = pd.read_csv(f"data/processed/{ten_tap}.csv")
    df["text"] = df["text"].astype(str)
    df["split"] = ten_tap
    df["y"] = (df["label"] == "spam").astype(int)
    df["p_spam"] = xac_suat_spam(df["text"].tolist())
    df["pred"] = (df["p_spam"] >= 0.5).astype(int)
    df["loi"] = df["pred"] != df["y"]
    df["fp"] = (df["pred"] == 1) & (df["y"] == 0)
    df["fn"] = (df["pred"] == 0) & (df["y"] == 1)

    s = df["text"]
    df["do_dai"] = s.str.len()
    df["co_day_so_dai"] = s.str.contains(r"\d{5,}")  # so dien thoai, ma dich vu
    df["co_url"] = s.str.contains(r"(?i)www\.|http|\.com|\.co\.uk")
    df["co_ky_hieu_tien"] = s.str.contains(r"[£$€]")
    ti_le_hoa = s.apply(
        lambda t: sum(c.isupper() for c in t) / max(1, sum(c.isalpha() for c in t)))
    df["nhieu_chu_hoa"] = ti_le_hoa > 0.5
    return df


def loai_loi(r):
    if not r["loi"]:
        return ""
    if r["y"] == 0:
        return "FP-chac chan" if r["p_spam"] > 0.9 else "FP-luong lu"
    return "FN-chac chan" if r["p_spam"] < 0.1 else "FN-luong lu"


def thong_ke_nhom(df, cot):
    g = df.groupby(cot, observed=True).agg(
        so_tin=("loi", "size"), so_loi=("loi", "sum"),
        FP=("fp", "sum"), FN=("fn", "sum"))
    g["ty_le_loi_%"] = (g["so_loi"] / g["so_tin"] * 100).round(1)
    return g


du_lieu = pd.concat([chuan_bi("val"), chuan_bi("test")], ignore_index=True)
du_lieu["loai_loi"] = du_lieu.apply(loai_loi, axis=1)

print("=== A. TONG QUAN LOI (nguong 0.5) ===")
print(du_lieu.groupby("split").agg(
    so_tin=("loi", "size"), so_loi=("loi", "sum"),
    FP=("fp", "sum"), FN=("fn", "sum")).to_string())

print("\n=== B. LOI 'CHAC CHAN' hay 'LUONG LU' ===")
print(du_lieu[du_lieu["loi"]].groupby(["split", "loai_loi"]).size().to_string())

print("\n=== C. TY LE LOI THEO DO DAI TIN (val + test gop) ===")
du_lieu["nhom_do_dai"] = pd.cut(
    du_lieu["do_dai"], bins=[0, 40, 100, 160, 100000],
    labels=["<=40 ky tu", "41-100", "101-160", ">160"])
print(thong_ke_nhom(du_lieu, "nhom_do_dai").to_string())

print("\n=== D. TY LE LOI THEO DAC DIEM TIN (val + test gop) ===")
for cot in ["co_day_so_dai", "co_url", "co_ky_hieu_tien", "nhieu_chu_hoa"]:
    print(f"\n-- {cot} --")
    print(thong_ke_nhom(du_lieu, cot).to_string())

print("\n=== E. DANH DOI THEO NGUONG (CHI DUNG TAP VAL) ===")
v = du_lieu[du_lieu["split"] == "val"]
dong = []
for nguong in np.arange(0.1, 0.95, 0.1):
    pred = (v["p_spam"] >= nguong).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        v["y"], pred, pos_label=1, average="binary", zero_division=0)
    dong.append({
        "nguong": round(float(nguong), 1),
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "FP": int(((pred == 1) & (v["y"] == 0)).sum()),
        "FN": int(((pred == 0) & (v["y"] == 1)).sum()),
    })
print(pd.DataFrame(dong).to_string(index=False))

print("\n=== F. CAC TIN SAI (chi de MO TA, khong dung de chinh mo hinh) ===")
loi = du_lieu[du_lieu["loi"]].sort_values(["split", "p_spam"])
for _, r in loi.iterrows():
    print(f"[{r['split']}] that={r['label']} p_spam={r['p_spam']:.3f} "
          f"{r['loai_loi']} | {r['text'][:140]}")

cot_luu = ["split", "label", "p_spam", "loai_loi", "do_dai", "co_day_so_dai",
           "co_url", "co_ky_hieu_tien", "nhieu_chu_hoa", "text"]
loi[cot_luu].to_csv("reports/loi_chi_tiet.csv", index=False, encoding="utf-8-sig")
print(f"\nDa luu {len(loi)} tin sai vao reports/loi_chi_tiet.csv")