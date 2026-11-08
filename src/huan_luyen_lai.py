"""Huan luyen lai DistilBERT voi du lieu tieng Viet bo sung (mo hinh ung vien).

Quy tac da khai bao truoc khi chay:
- Giu nguyen sieu tham so mo hinh hien tai: SEED=42, MAX_LEN=64, BATCH=16, EPOCHS=3, LR=2e-5.
- Train = UCI train + 90 tin tieng Viet. Chon epoch theo F1 tren (UCI val + 30 tin vi val).
- UCI test va bo do co dinh 50 tin chi do MOT LAN o cuoi, khong dung de chon gi.
- Khong ghi de mo hinh dang chay (models/distilbert); ung vien luu o models/distilbert_vi.
- Ghi truc tiep vao MLflow (tag nguon = chay_truc_tiep).
"""
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOC = Path(__file__).resolve().parent.parent

# --- Sieu tham so: GIU NGUYEN nhu train_distilbert.py ---
SEED = 42
CKPT = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 16
EPOCHS = 3
LR = 2e-5
SO_TIN_VI_VAL = 30

# --- Quy tac khai bao truoc khi chay ---
NGUONG_SO_SPAM_VI_BAT_DUOC = 5   # tren bo do 50 tin: bat duoc >= 5/11 spam tieng Viet
GIAM_F1_TEST_TOI_DA = 0.01       # F1 UCI test khong giam qua 0.01 so voi mo hinh hien tai
FP_CO_SO_BO_DO = 2               # mo hinh hien tai chan nham 2 tin tren bo do (buoc 6c-2b)
TANG_FP_TOI_DA = 2               # FP tren bo do khong tang qua 2

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEN_THI_NGHIEM = "huan_luyen_lai_phan_hoi"
THU_MUC_UNG_VIEN = GOC / "models" / "distilbert_vi"
TEP_KET_QUA = GOC / "reports" / "ket_qua_huan_luyen_lai.json"
TEP_VI = GOC / "data" / "feedback" / "tin_train_vi.csv"
TEP_DO = GOC / "data" / "feedback" / "bo_do_co_dinh_50_tin.csv"
TEP_KIEM_TRUNG = GOC / "reports" / "kiem_trung_lap_tom_tat.txt"
TEP_TEST_CU = GOC / "reports" / "ket_qua_test.csv"


def dung(thong_bao):
    sys.exit(thong_bao)


# ---------- Chot chan truoc khi chay ----------
if THU_MUC_UNG_VIEN.exists() or TEP_KET_QUA.exists():
    dung("Da co models/distilbert_vi hoac reports/ket_qua_huan_luyen_lai.json. "
         "Khong chay lai de tranh do lai tren bo do. Hay bao lai de duoc huong dan.")
for tep in (TEP_VI, TEP_DO, TEP_KIEM_TRUNG, TEP_TEST_CU):
    if not tep.exists():
        dung(f"Khong thay tep can thiet: {tep}")

bao_cao = {}
for dong in TEP_KIEM_TRUNG.read_text(encoding="utf-8").splitlines():
    if "=" in dong:
        k, v = dong.split("=", 1)
        bao_cao[k.strip()] = v.strip()
if bao_cao.get("so_tin_vuot_nguong") != "0" or bao_cao.get("trung_y_het_bo_do") != "0":
    dung("Chot chan ro ri chua dat. Chay python src\\kiem_trung_lap.py va sua truoc.")

try:
    urllib.request.urlopen(MLFLOW_URI + "/health", timeout=3)
except Exception:
    dung(f"Khong ket noi duoc may chu MLflow tai {MLFLOW_URI}. Hay bat server truoc.")

kt_cu = pd.read_csv(TEP_TEST_CU)
hang_cu = kt_cu[kt_cu["model"].str.contains("DistilBERT")]
if len(hang_cu) == 0:
    dung("Khong thay dong DistilBERT trong reports/ket_qua_test.csv")
F1_TEST_CO_SO = float(hang_cu["f1_spam"].iloc[0])

# ---------- Hat giong ngau nhien va thiet bi ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiet bi:", device)

# ---------- Doc du lieu ----------
uci_train = pd.read_csv(GOC / "data" / "processed" / "train.csv")
uci_val = pd.read_csv(GOC / "data" / "processed" / "val.csv")
uci_test = pd.read_csv(GOC / "data" / "processed" / "test.csv")
vi = pd.read_csv(TEP_VI)
do = pd.read_csv(TEP_DO)
for df in (vi, do):
    df["label"] = df["label"].astype(str).str.strip().str.lower()


def chuan(s):
    return " ".join(str(s).lower().split())


# Khong cho phep tin train trung y het val/test/bo do
cam = set(map(chuan, do["text"])) | set(map(chuan, uci_val["text"])) | set(map(chuan, uci_test["text"]))
if any(chuan(t) in cam for t in vi["text"]):
    dung("Co tin train tieng Viet trung y het val/test/bo do. Dung lai.")

# Chia 120 tin tieng Viet thanh 90 train / 30 val, giu ti le nhan, seed co dinh
vi_train, vi_val = train_test_split(
    vi, test_size=SO_TIN_VI_VAL, stratify=vi["label"], random_state=SEED)
vi_train.to_csv(GOC / "data" / "feedback" / "tin_train_vi_train.csv", index=False)
vi_val.to_csv(GOC / "data" / "feedback" / "tin_train_vi_val.csv", index=False)

train = pd.concat([uci_train[["text", "label"]], vi_train[["text", "label"]]], ignore_index=True)
print(f"Train: {len(train)} tin (UCI {len(uci_train)} + tieng Viet {len(vi_train)})")
print(f"Val tieng Viet: {len(vi_val)} tin, spam {int((vi_val['label'] == 'spam').sum())}")

tokenizer = AutoTokenizer.from_pretrained(CKPT)


def to_tensors(df):
    enc = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    y = torch.tensor((df["label"] == "spam").astype(int).to_numpy())
    return enc["input_ids"], enc["attention_mask"], y


ids_tr, mask_tr, ytr = to_tensors(train)
t_uci_val = to_tensors(uci_val)
t_vi_val = to_tensors(vi_val)
t_uci_test = to_tensors(uci_test)
t_do = to_tensors(do)

model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2)
model.to(device)

n_spam = int(ytr.sum())
n_ham = len(ytr) - n_spam
weights = torch.tensor([1.0, n_ham / n_spam], dtype=torch.float32, device=device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def predict(ids, mask):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            out = model(
                input_ids=ids[i:i + 64].to(device),
                attention_mask=mask[i:i + 64].to(device),
            )
            preds.append(out.logits.argmax(dim=-1).cpu())
    return torch.cat(preds).numpy()


def danh_gia(pred, y_true):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label=1, average="binary", zero_division=0)
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "fp": fp, "fn": fn}


def danh_gia_val():
    """Tra ve (UCI val, vi val, gop). Chon epoch theo F1 cua tap gop."""
    pred_u = predict(t_uci_val[0], t_uci_val[1])
    pred_v = predict(t_vi_val[0], t_vi_val[1])
    y_u = t_uci_val[2].numpy()
    y_v = t_vi_val[2].numpy()
    gop = danh_gia(np.concatenate([pred_u, pred_v]), np.concatenate([y_u, y_v]))
    return danh_gia(pred_u, y_u), danh_gia(pred_v, y_v), gop


mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(TEN_THI_NGHIEM)

with mlflow.start_run(run_name="DistilBERT + 90 tin tieng Viet (ung vien)"):
    mlflow.set_tags({
        "nguon": "chay_truc_tiep",
        "ten_mo_hinh": "DistilBERT ung vien (them 90 tin vi)",
        "tap_du_lieu_train": "UCI train + tin_train_vi (90)",
    })
    mlflow.log_params({
        "seed": SEED, "ckpt": CKPT, "max_len": MAX_LEN, "batch": BATCH,
        "epochs": EPOCHS, "lr": LR, "weight_decay": 0.01,
        "so_tin_train_uci": len(uci_train), "so_tin_train_vi": len(vi_train),
        "so_tin_val_vi": len(vi_val), "cach_chon_epoch": "F1 tren UCI val + vi val",
        "nguong_spam_vi_bat_duoc": NGUONG_SO_SPAM_VI_BAT_DUOC,
        "giam_f1_test_toi_da": GIAM_F1_TEST_TOI_DA,
        "tang_fp_bo_do_toi_da": TANG_FP_TOI_DA,
    })

    best_f1, best_state, best_epoch = -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(ids_tr))
        total = 0.0
        for i in range(0, len(ids_tr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = model(
                input_ids=ids_tr[idx].to(device),
                attention_mask=mask_tr[idx].to(device),
            )
            loss = loss_fn(out.logits, ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        m_u, m_v, m_g = danh_gia_val()
        print(f"Epoch {epoch} | loss {total / len(ids_tr):.4f} | "
              f"UCI val F1 {m_u['f1']:.3f} | vi val F1 {m_v['f1']:.3f} "
              f"(R {m_v['recall']:.3f}) | gop F1 {m_g['f1']:.3f}")
        mlflow.log_metric("train_loss", total / len(ids_tr), step=epoch)
        mlflow.log_metric("val_uci_f1", m_u["f1"], step=epoch)
        mlflow.log_metric("val_vi_f1", m_v["f1"], step=epoch)
        mlflow.log_metric("val_vi_recall", m_v["recall"], step=epoch)
        mlflow.log_metric("val_gop_f1", m_g["f1"], step=epoch)
        if m_g["f1"] > best_f1:
            best_f1, best_epoch = m_g["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best_state)
    print(f"Epoch tot nhat (theo F1 gop): {best_epoch}")

    # --- Do cuoi, mot lan duy nhat ---
    m_u, m_v, m_g = danh_gia_val()
    pred_test = predict(t_uci_test[0], t_uci_test[1])
    m_test = danh_gia(pred_test, t_uci_test[2].numpy())

    pred_do = predict(t_do[0], t_do[1])
    y_do = t_do[2].numpy()
    ngon = do["ngon_ngu"].astype(str).str.strip().str.lower().to_numpy()
    m_do = danh_gia(pred_do, y_do)
    ket_qua_do = {}
    for nn_ in ("vi", "en"):
        spam_that = (y_do == 1) & (ngon == nn_)
        ket_qua_do[nn_] = {
            "so_spam_that": int(spam_that.sum()),
            "so_spam_bat_duoc": int((pred_do[spam_that] == 1).sum()),
            "so_ham_bi_chan_nham": int(((pred_do == 1) & (y_do == 0) & (ngon == nn_)).sum()),
        }

    dk_a = ket_qua_do["vi"]["so_spam_bat_duoc"] >= NGUONG_SO_SPAM_VI_BAT_DUOC
    dk_b = m_test["f1"] >= F1_TEST_CO_SO - GIAM_F1_TEST_TOI_DA
    dk_c = m_do["fp"] <= FP_CO_SO_BO_DO + TANG_FP_TOI_DA

    # Luu ung vien o thu muc rieng (models/ da bi .gitignore)
    THU_MUC_UNG_VIEN.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(THU_MUC_UNG_VIEN)
    tokenizer.save_pretrained(THU_MUC_UNG_VIEN)

    ket_qua = {
        "epoch_tot_nhat": best_epoch,
        "thoi_gian_hoc_giay": round(train_seconds, 1),
        "val_uci": m_u, "val_vi": m_v, "val_gop": m_g,
        "test_uci": m_test, "f1_test_co_so": F1_TEST_CO_SO,
        "bo_do_50_tong": m_do, "bo_do_50_theo_ngon_ngu": ket_qua_do,
        "dieu_kien": {
            "a_bat_duoc_toi_thieu_spam_vi": bool(dk_a),
            "b_f1_test_khong_giam_qua_muc": bool(dk_b),
            "c_fp_bo_do_khong_tang_qua_muc": bool(dk_c),
        },
    }
    TEP_KET_QUA.write_text(json.dumps(ket_qua, indent=2, ensure_ascii=True), encoding="utf-8")

    for ten, gia_tri in (
        ("val_uci_f1_cuoi", m_u["f1"]), ("val_vi_f1_cuoi", m_v["f1"]),
        ("val_vi_recall_cuoi", m_v["recall"]),
        ("test_uci_precision", m_test["precision"]), ("test_uci_recall", m_test["recall"]),
        ("test_uci_f1", m_test["f1"]), ("test_uci_fp", m_test["fp"]), ("test_uci_fn", m_test["fn"]),
        ("do50_f1", m_do["f1"]), ("do50_fp", m_do["fp"]), ("do50_fn", m_do["fn"]),
        ("do50_spam_vi_bat_duoc", ket_qua_do["vi"]["so_spam_bat_duoc"]),
        ("do50_spam_en_bat_duoc", ket_qua_do["en"]["so_spam_bat_duoc"]),
        ("epoch_tot_nhat", best_epoch), ("thoi_gian_hoc_giay", train_seconds),
    ):
        mlflow.log_metric(ten, float(gia_tri))
    mlflow.log_artifact(str(TEP_KET_QUA))

print("\n=== KET QUA UNG VIEN (do mot lan) ===")
print(f"UCI test: P {m_test['precision']:.3f} R {m_test['recall']:.3f} F1 {m_test['f1']:.3f} "
      f"FP {m_test['fp']} FN {m_test['fn']}  (co so F1 {F1_TEST_CO_SO:.3f})")
print(f"Bo do 50 tin - tieng Viet: bat duoc {ket_qua_do['vi']['so_spam_bat_duoc']}/"
      f"{ket_qua_do['vi']['so_spam_that']} spam, chan nham {ket_qua_do['vi']['so_ham_bi_chan_nham']} ham")
print(f"Bo do 50 tin - tieng Anh: bat duoc {ket_qua_do['en']['so_spam_bat_duoc']}/"
      f"{ket_qua_do['en']['so_spam_that']} spam, chan nham {ket_qua_do['en']['so_ham_bi_chan_nham']} ham")
print(f"Bo do 50 tin - tong: FP {m_do['fp']} FN {m_do['fn']}  (co so FP {FP_CO_SO_BO_DO})")
print("\n=== DOI CHIEU QUY TAC DA KHAI BAO ===")
print(f"(a) bat duoc >= {NGUONG_SO_SPAM_VI_BAT_DUOC}/11 spam tieng Viet: {'DAT' if dk_a else 'KHONG DAT'}")
print(f"(b) F1 UCI test >= {F1_TEST_CO_SO - GIAM_F1_TEST_TOI_DA:.3f}: {'DAT' if dk_b else 'KHONG DAT'}")
print(f"(c) FP bo do <= {FP_CO_SO_BO_DO + TANG_FP_TOI_DA}: {'DAT' if dk_c else 'KHONG DAT'}")
print("Ket luan thay the hay khong se quyet dinh o buoc 6c-4, khong tu dong thay.")
print(f"\nDa luu ung vien o {THU_MUC_UNG_VIEN} va ket qua o {TEP_KET_QUA}")
print(f"Xem MLflow: {MLFLOW_URI}")