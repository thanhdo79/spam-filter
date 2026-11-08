"""Dang ky mo hinh vao MLflow Model Registry.

- champion  = mo hinh dang chay (models/distilbert), ban goc de xuat ONNX.
- challenger = ung vien buoc 6c-3b (models/distilbert_vi), da bi tu choi theo quy tac.
Chi chay mot lan (co chot chan). Khong thay doi mo hinh API dang phuc vu.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

GOC = Path(__file__).resolve().parent.parent
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEN_THI_NGHIEM = "dang_ky_mo_hinh"
TEN_MO_HINH = "spam_distilbert"
THU_MUC_CHAMPION = GOC / "models" / "distilbert"
THU_MUC_CHALLENGER = GOC / "models" / "distilbert_vi"
TEP_KET_QUA = GOC / "reports" / "ket_qua_huan_luyen_lai.json"
TEP_TEST_CU = GOC / "reports" / "ket_qua_test.csv"

# So lieu cua mo hinh hien tai tren bo do 50 tin, lay tu buoc 6c-2b
CHAMPION_DO50_FP = 2
CHAMPION_DO50_SPAM_VI = "0/11"
CHAMPION_DO50_SPAM_EN = "10/11"

for thu_muc in (THU_MUC_CHAMPION, THU_MUC_CHALLENGER):
    if not thu_muc.exists():
        sys.exit(f"Khong thay thu muc mo hinh: {thu_muc}")
for tep in (TEP_KET_QUA, TEP_TEST_CU):
    if not tep.exists():
        sys.exit(f"Khong thay tep: {tep}")

try:
    urllib.request.urlopen(MLFLOW_URI + "/health", timeout=3)
except Exception:
    sys.exit(f"Khong ket noi duoc MLflow tai {MLFLOW_URI}. Hay bat server truoc.")

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# chot chan: da dang ky roi thi dung lai
if client.search_registered_models(filter_string=f"name = '{TEN_MO_HINH}'"):
    sys.exit(f"Mo hinh '{TEN_MO_HINH}' da co trong Registry. Khong dang ky lai de tranh trung.")

client.create_registered_model(
    TEN_MO_HINH,
    description="Bo loc spam DistilBERT. Alias champion = mo hinh dang chay, challenger = ung vien.",
)
mlflow.set_experiment(TEN_THI_NGHIEM)

kq = json.loads(TEP_KET_QUA.read_text(encoding="utf-8"))
kt_cu = pd.read_csv(TEP_TEST_CU)
f1_test_champion = float(kt_cu[kt_cu["model"].str.contains("DistilBERT")]["f1_spam"].iloc[0])
vi = kq["bo_do_50_theo_ngon_ngu"]["vi"]
en = kq["bo_do_50_theo_ngon_ngu"]["en"]


def dang_ky(ten_run, thu_muc, mo_ta, tags_phien_ban):
    """Ghi thu muc mo hinh thanh artifact cua mot run roi dang ky thanh phien ban moi."""
    with mlflow.start_run(run_name=ten_run) as run:
        mlflow.set_tags({"nguon": "dang_ky_registry", "thu_muc_goc": str(thu_muc.name)})
        mlflow.log_artifacts(str(thu_muc), artifact_path="mo_hinh")
        uri = f"runs:/{run.info.run_id}/mo_hinh"
    # MLflow 3 khong cho register_model tren thu muc artifact tu do (khong phai logged model),
    # nen tao phien ban truc tiep bang client, tro toi thu muc artifact cua run.
    mv = client.create_model_version(TEN_MO_HINH, source=uri, run_id=run.info.run_id)
    client.update_model_version(TEN_MO_HINH, mv.version, description=mo_ta)
    for k, v in tags_phien_ban.items():
        client.set_model_version_tag(TEN_MO_HINH, mv.version, k, str(v))
    return mv.version


v_champion = dang_ky(
    "dang ky champion (mo hinh dang chay)",
    THU_MUC_CHAMPION,
    "Mo hinh dang chay: DistilBERT tinh chinh tren UCI. Ban goc de xuat ONNX phuc vu API.",
    {
        "f1_test_uci": f"{f1_test_champion:.3f}",
        "do50_spam_vi_bat_duoc": CHAMPION_DO50_SPAM_VI,
        "do50_spam_en_bat_duoc": CHAMPION_DO50_SPAM_EN,
        "do50_fp": CHAMPION_DO50_FP,
        "ghi_chu": "dang phuc vu",
    },
)
v_challenger = dang_ky(
    "dang ky challenger (ung vien 6c-3b)",
    THU_MUC_CHALLENGER,
    "Ung vien: DistilBERT huan luyen lai voi UCI train + 90 tin tieng Viet. "
    "Bi tu choi theo quy tac (dieu kien c: FP bo do 9 > 4). Xem docs/quy_tac_thay_mo_hinh.md.",
    {
        "f1_test_uci": f"{kq['test_uci']['f1']:.3f}",
        "do50_spam_vi_bat_duoc": f"{vi['so_spam_bat_duoc']}/{vi['so_spam_that']}",
        "do50_spam_en_bat_duoc": f"{en['so_spam_bat_duoc']}/{en['so_spam_that']}",
        "do50_fp": kq["bo_do_50_tong"]["fp"],
        "dieu_kien_a": kq["dieu_kien"]["a_bat_duoc_toi_thieu_spam_vi"],
        "dieu_kien_b": kq["dieu_kien"]["b_f1_test_khong_giam_qua_muc"],
        "dieu_kien_c": kq["dieu_kien"]["c_fp_bo_do_khong_tang_qua_muc"],
        "ghi_chu": "bi tu choi theo quy tac",
    },
)

client.set_registered_model_alias(TEN_MO_HINH, "champion", v_champion)
client.set_registered_model_alias(TEN_MO_HINH, "challenger", v_challenger)

print(f"Da dang ky '{TEN_MO_HINH}': phien ban {v_champion} = champion, phien ban {v_challenger} = challenger.")
print(f"Xem tai: {MLFLOW_URI}/#/models/{TEN_MO_HINH}")