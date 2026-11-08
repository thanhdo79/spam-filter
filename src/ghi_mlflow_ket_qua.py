"""Nhap ket qua so sanh mo hinh da co (CSV) vao MLflow de theo doi thi nghiem.

Chi chay mot lan. Co chot chan de khong nhap trung.
"""
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

GOC = Path(__file__).resolve().parent.parent
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEN_THI_NGHIEM = "so_sanh_mo_hinh_spam"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(TEN_THI_NGHIEM)

# chot chan: neu da nhap tu CSV roi thi dung lai, khong nhap trung
da_co = mlflow.search_runs(
    experiment_names=[TEN_THI_NGHIEM],
    filter_string="tags.nguon = 'nhap_tu_csv'",
)
if len(da_co) > 0:
    sys.exit(f"Da co {len(da_co)} run nhap tu CSV. Khong nhap lai de tranh trung.")


def nhap(tep, tap_du_lieu):
    """Moi dong trong CSV thanh mot run trong MLflow."""
    duong_dan = GOC / "reports" / tep
    df = pd.read_csv(duong_dan)
    so_run = 0
    for _, hang in df.iterrows():
        ten = str(hang["model"])
        with mlflow.start_run(run_name=f"{ten} | {tap_du_lieu}"):
            mlflow.set_tags({
                "tap_du_lieu": tap_du_lieu,
                "ten_mo_hinh": ten,
                "nguon": "nhap_tu_csv",
            })
            mlflow.log_param("du_lieu", tap_du_lieu)
            for cot, gia_tri in hang.items():
                if cot == "model" or pd.isna(gia_tri):
                    continue
                try:
                    mlflow.log_metric(cot, float(gia_tri))
                except (TypeError, ValueError):
                    pass  # bo qua cot khong phai so
            mlflow.log_artifact(str(duong_dan))
        so_run += 1
    return so_run


n_val = nhap("comparison_val.csv", "val")
n_test = nhap("ket_qua_test.csv", "test")
print(f"Da nhap {n_val} run (val) va {n_test} run (test) vao '{TEN_THI_NGHIEM}'.")
print(f"Xem tai: {MLFLOW_URI}")