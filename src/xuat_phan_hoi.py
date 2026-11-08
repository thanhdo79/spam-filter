import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DUONG_DAN_DB = Path("data/docker/app.db")
THU_MUC_RA = Path("data/feedback")
TEP_SACH = THU_MUC_RA / "phan_hoi_sach.csv"
TEP_TOM_TAT = Path("reports/phan_hoi_tom_tat.txt")

DONG_KET_QUA = []


def in_ra(*args):
    """In ra man hinh va luu lai de ghi vao tep tom tat."""
    dong = " ".join(str(a) for a in args)
    print(dong)
    DONG_KET_QUA.append(dong)


def chuan_hoa(chuoi):
    """Chuan hoa de so sanh: chu thuong, bo khoang trang thua."""
    return " ".join(str(chuoi).lower().split())


if not DUONG_DAN_DB.exists():
    sys.exit(f"Khong tim thay co so du lieu: {DUONG_DAN_DB}")

# ---------- 1. Doc cac tin da co phan hoi ----------
conn = sqlite3.connect(DUONG_DAN_DB)
pb = pd.read_sql_query(
    "SELECT id, text, label, nhan_dung FROM du_doan "
    "WHERE nhan_dung IS NOT NULL ORDER BY id",
    conn,
)
conn.close()

in_ra("=== TOM TAT PHAN HOI NGUOI DUNG ===")
in_ra(f"Tong so tin co phan hoi: {len(pb)}")
if len(pb) > 0:
    so_sai = int((pb["label"] != pb["nhan_dung"]).sum())
    in_ra(f"Mo hinh du doan dung: {len(pb) - so_sai} | du doan sai: {so_sai}")
    # Noi dung tin chi in ra man hinh, KHONG ghi vao tep tom tat (rieng tu)
    print("\nCac tin co phan hoi (toi da 10 tin dau, chi hien tren man hinh):")
    for _, r in pb.head(10).iterrows():
        print(f"  id={r['id']} | mo hinh: {r['label']} | "
              f"nguoi dung: {r['nhan_dung']} | {r['text'][:60]!r}")

pb["khoa"] = pb["text"].map(chuan_hoa)

# ---------- 2. Xung dot nhan: cung noi dung nhung gan nhan khac nhau ----------
so_nhan = pb.groupby("khoa")["nhan_dung"].nunique()
khoa_xung_dot = set(so_nhan[so_nhan > 1].index)
pb_ok = pb[~pb["khoa"].isin(khoa_xung_dot)]
in_ra(f"\nSo noi dung bi gan nhan xung dot (bi loai): {len(khoa_xung_dot)}")

# ---------- 3. Bo trung lap noi bo (giu phan hoi moi nhat) ----------
truoc = len(pb_ok)
pb_ok = pb_ok.drop_duplicates(subset="khoa", keep="last")
in_ra(f"So dong trung lap noi bo bi bo: {truoc - len(pb_ok)}")

# ---------- 4. Chong ro ri: loai tin da co trong train / val / test ----------
tap = {}
for ten in ("train", "val", "test"):
    df = pd.read_csv(f"data/processed/{ten}.csv")
    tap[ten] = set(df["text"].astype(str).map(chuan_hoa))

trung_train = pb_ok["khoa"].isin(tap["train"])
trung_val = pb_ok["khoa"].isin(tap["val"])
trung_test = pb_ok["khoa"].isin(tap["test"])
in_ra(f"Trung voi train (khong co thong tin moi): {int(trung_train.sum())}")
in_ra(f"Trung voi val (cam dung de huan luyen): {int(trung_val.sum())}")
in_ra(f"Trung voi test (cam dung de huan luyen): {int(trung_test.sum())}")

con_lai = pb_ok[~(trung_train | trung_val | trung_test)]
in_ra(f"\nSo tin sach con lai co the dung de huan luyen: {len(con_lai)}")
if len(con_lai) > 0:
    dem = con_lai["nhan_dung"].value_counts().to_dict()
    in_ra(f"Phan bo nhan: {dem}")

# ---------- 5. Ghi tep ----------
THU_MUC_RA.mkdir(parents=True, exist_ok=True)
xuat = con_lai[["text", "nhan_dung"]].rename(columns={"nhan_dung": "label"})
xuat.to_csv(TEP_SACH, index=False, encoding="utf-8")
Path("reports").mkdir(exist_ok=True)
with open(TEP_TOM_TAT, "w", encoding="utf-8") as f:
    f.write("\n".join(DONG_KET_QUA) + "\n")
print(f"\nDa ghi {TEP_SACH} (khong dua len GitHub) va {TEP_TOM_TAT}")