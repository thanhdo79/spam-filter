"""Kiem tra tin train tieng Viet co qua giong bo do co dinh hay khong (nguy co ro ri).

Dung do tuong dong cosine tren n-gram ky tu (khong dau, chu thuong).
Chi in ra man hinh noi dung tin giong nhau; tep bao cao chi chua so dem.
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GOC = Path(__file__).resolve().parent.parent
THU_MUC = GOC / "data" / "feedback"
TEP_TRAIN = THU_MUC / "tin_train_vi.csv"
TEP_DO = THU_MUC / "bo_do_co_dinh_50_tin.csv"
NGUONG = 0.5  # nguong chon thu cong, khong phai quy tac thong ke


def chuan_hoa(s):
    """Chu thuong, bo dau tieng Viet, gom khoang trang."""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d")
    return " ".join(s.split())


def doc(tep):
    if not tep.exists():
        sys.exit(f"Khong thay tep: {tep}")
    df = pd.read_csv(tep)
    if not {"text", "label"}.issubset(df.columns):
        sys.exit(f"Tep {tep.name} thieu cot text/label.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["nhan"] = df["label"].str.strip().str.lower()
    loi = df[~df["nhan"].isin(["spam", "ham"])]
    if len(loi) > 0:
        sys.exit(f"Tep {tep.name} co nhan khong hop le (chi nhan spam/ham).")
    df["chuan"] = df["text"].map(chuan_hoa)
    return df.reset_index(drop=True)


train = doc(TEP_TRAIN)
do = doc(TEP_DO)

trung_y_het = int(train["chuan"].isin(set(do["chuan"])).sum())
trung_noi_bo = int(train["chuan"].duplicated().sum())

tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
tf.fit(pd.concat([train["chuan"], do["chuan"]]))
sim = cosine_similarity(tf.transform(train["chuan"]), tf.transform(do["chuan"]))
sim_max = sim.max(axis=1)
vi_tri_do = sim.argmax(axis=1)

qua_giong = [i for i in range(len(train)) if sim_max[i] >= NGUONG]

print("=== KIEM TRA RO RI TRAIN -> BO DO CO DINH ===")
print(f"So tin train: {len(train)} | phan bo nhan: {train['nhan'].value_counts().to_dict()}")
print(f"Trung y het bo do (sau chuan hoa): {trung_y_het}")
print(f"Trung lap noi bo trong train: {trung_noi_bo}")
print(f"Do giong lon nhat (trung binh): {sim_max.mean():.3f} | lon nhat: {sim_max.max():.3f}")
print(f"So tin train co do giong >= {NGUONG}: {len(qua_giong)}")
for i in qua_giong[:15]:
    j = vi_tri_do[i]
    print(f"  sim={sim_max[i]:.2f}")
    print(f"    train: {train['text'][i][:70]}")
    print(f"    do   : {do['text'][j][:70]}")

with open(GOC / "reports" / "kiem_trung_lap_tom_tat.txt", "w", encoding="utf-8") as f:
    f.write(f"so_tin_train={len(train)}\n")
    f.write(f"trung_y_het_bo_do={trung_y_het}\n")
    f.write(f"trung_noi_bo={trung_noi_bo}\n")
    f.write(f"nguong_giong={NGUONG}\n")
    f.write(f"so_tin_vuot_nguong={len(qua_giong)}\n")
    f.write(f"do_giong_trung_binh={sim_max.mean():.3f}\n")
    f.write(f"do_giong_lon_nhat={sim_max.max():.3f}\n")

print("\nDa ghi reports\\kiem_trung_lap_tom_tat.txt")
if trung_y_het > 0 or qua_giong:
    print("CANH BAO: sua hoac bo cac tin tren truoc khi huan luyen.")
    sys.exit(1)
print("DAT: khong thay tin nao qua giong bo do co dinh.")