import sys
from pathlib import Path

import httpx
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIA_CHI_API = "http://127.0.0.1:8000"
TEP_VAO = Path("data/feedback/tin_nhom_tao.csv")
TEP_KET_QUA = Path("data/feedback/ket_qua_gui.csv")
TEP_TOM_TAT = Path("reports/phan_hoi_nhom_tom_tat.txt")
# Chay "python src\gui_phan_hoi_nhom.py kiem-tra" de chi kiem tra du lieu, khong gui
CHI_KIEM_TRA = len(sys.argv) > 1 and sys.argv[1] == "kiem-tra"

DONG_KET_QUA = []


def in_ra(*args):
    """In ra man hinh va luu lai de ghi vao tep tom tat."""
    dong = " ".join(str(a) for a in args)
    print(dong)
    DONG_KET_QUA.append(dong)


# ---------- 1. Doc va kiem tra tep dau vao ----------
if not TEP_VAO.exists():
    sys.exit(f"Khong tim thay {TEP_VAO}")
if TEP_KET_QUA.exists() and not CHI_KIEM_TRA:
    sys.exit("Da gui roi (co tep ket_qua_gui.csv). Neu xoa tep nay de gui lai, "
             "co so du lieu se co tin trung lap.")

df = pd.read_csv(TEP_VAO, encoding="utf-8")
cot_can = {"text", "label", "ngon_ngu"}
if not cot_can.issubset(df.columns):
    sys.exit(f"Thieu cot, can co {sorted(cot_can)}; dang co {list(df.columns)}")
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["ngon_ngu"] = df["ngon_ngu"].astype(str).str.strip().str.lower()

loi = []
if (df["text"].eq("") | df["text"].eq("nan")).any():
    loi.append("co dong text rong")
if (~df["label"].isin(["spam", "ham"])).any():
    loi.append("co label khong phai spam hoac ham")
if (~df["ngon_ngu"].isin(["vi", "en"])).any():
    loi.append("co ngon_ngu khong phai vi hoac en")
if (df["text"].str.len() > 2000).any():
    loi.append("co tin dai hon 2000 ky tu")
so_trung = int(df["text"].str.lower().duplicated().sum())
if so_trung:
    loi.append(f"co {so_trung} dong trung noi dung")
if loi:
    sys.exit("Loi du lieu dau vao: " + "; ".join(loi))

print(f"Tong so tin: {len(df)}")
print(df.groupby(["ngon_ngu", "label"]).size().to_string())
if CHI_KIEM_TRA:
    sys.exit("\nChe do kiem tra: du lieu hop le, chua gui gi.")

# ---------- 2. Gui tung tin qua /predict roi /feedback ----------
ket_qua = []
with httpx.Client(timeout=30) as client:
    client.get(f"{DIA_CHI_API}/health").raise_for_status()
    for _, r in df.iterrows():
        p = client.post(f"{DIA_CHI_API}/predict", json={"text": r["text"]})
        p.raise_for_status()
        du_doan = p.json()
        f = client.post(f"{DIA_CHI_API}/feedback",
                        json={"id": du_doan["id"], "correct_label": r["label"]})
        f.raise_for_status()
        ket_qua.append({
            "id": du_doan["id"], "ngon_ngu": r["ngon_ngu"], "that": r["label"],
            "doan": du_doan["label"], "p_spam": du_doan["spam_probability"],
            "text": r["text"],
        })

kq = pd.DataFrame(ket_qua)
# Tep ket qua co noi dung tin nen chi luu trong data/feedback (khong len GitHub)
kq.to_csv(TEP_KET_QUA, index=False, encoding="utf-8")


# ---------- 3. Do mo hinh hien tai tren bo tin nay (moc "truoc") ----------
def tinh(ten, d):
    if len(d) == 0:
        return
    p, r, f1, _ = precision_recall_fscore_support(
        d["that"], d["doan"], pos_label="spam", average="binary",
        zero_division=0)
    fp = int(((d["doan"] == "spam") & (d["that"] == "ham")).sum())
    fn = int(((d["doan"] == "ham") & (d["that"] == "spam")).sum())
    dung = int((d["doan"] == d["that"]).sum())
    so_spam = int((d["that"] == "spam").sum())
    in_ra(f"{ten}: n={len(d)} (spam that={so_spam}) | dung {dung}/{len(d)} | "
          f"precision={p:.3f} recall={r:.3f} f1={f1:.3f} | FP={fp} FN={fn}")


in_ra("=== MO HINH HIEN TAI TREN BO TIN NHOM TAO (moc truoc huan luyen lai) ===")
tinh("Tat ca", kq)
tinh("Tieng Viet", kq[kq["ngon_ngu"] == "vi"])
tinh("Tieng Anh", kq[kq["ngon_ngu"] == "en"])

sai = kq[kq["that"] != kq["doan"]]
print(f"\nCac tin mo hinh doan sai ({len(sai)} tin, chi hien tren man hinh):")
for _, r in sai.iterrows():
    print(f"  id={r['id']} | {r['ngon_ngu']} | that: {r['that']} | "
          f"doan: {r['doan']} (p_spam={r['p_spam']}) | {r['text'][:60]!r}")

Path("reports").mkdir(exist_ok=True)
with open(TEP_TOM_TAT, "w", encoding="utf-8") as f:
    f.write("\n".join(DONG_KET_QUA) + "\n")
print(f"\nDa ghi {TEP_KET_QUA} (khong len GitHub) va {TEP_TOM_TAT}")