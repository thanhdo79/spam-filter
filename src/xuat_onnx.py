import os
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = "models/distilbert"
ONNX_PATH = "models/distilbert.onnx"
MAX_LEN = 64

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, attn_implementation="eager")
model.eval().cpu()


class BocLogits(torch.nn.Module):
    """Boc mo hinh de dau ra chi la ma tran logits."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


wrapper = BocLogits(model).eval()

mau = tok(
    ["Free entry in a weekly competition, text WIN to 80086",
     "Ok, see you at lunch tomorrow"],
    truncation=True, max_length=MAX_LEN, padding="max_length",
    return_tensors="pt",
)
args = (mau["input_ids"], mau["attention_mask"])
dynamic_axes = {
    "input_ids": {0: "batch"},
    "attention_mask": {0: "batch"},
    "logits": {0: "batch"},
}


def xuat(**kw):
    torch.onnx.export(
        wrapper, args, ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        **kw,
    )


os.makedirs("models", exist_ok=True)
try:
    xuat(dynamo=False)
    print("Da xuat bang bo xuat co dien (TorchScript)")
except Exception as e:
    print("Bo xuat co dien loi:", repr(e)[:300])
    print("Thu bo xuat moi (dynamo)...")
    xuat()
    print("Da xuat bang bo xuat moi (dynamo)")

print("Kich thuoc tep ONNX: %.1f MB" % (os.path.getsize(ONNX_PATH) / 1e6))

# ---------- Kiem tra: ONNX co cho ket qua giong PyTorch khong? ----------
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

val = pd.read_csv("data/processed/val.csv")
texts = val["text"].astype(str).tolist()


def ma_hoa(batch):
    return tok(batch, truncation=True, max_length=MAX_LEN,
               padding="max_length", return_tensors="np")


def chay_onnx(enc):
    return sess.run(None, {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0]


def chay_torch(enc):
    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(enc["input_ids"].astype(np.int64)),
            attention_mask=torch.from_numpy(
                enc["attention_mask"].astype(np.int64)),
        )
    return out.logits.numpy()


lo_torch, lo_onnx = [], []
for i in range(0, len(texts), 64):
    enc = ma_hoa(texts[i:i + 64])
    lo_torch.append(chay_torch(enc))
    lo_onnx.append(chay_onnx(enc))
lo_torch = np.concatenate(lo_torch)
lo_onnx = np.concatenate(lo_onnx)

max_diff = float(np.abs(lo_torch - lo_onnx).max())
giong_nhan = float((lo_torch.argmax(1) == lo_onnx.argmax(1)).mean())
print(f"\nSo tin kiem tra: {len(texts)}")
print(f"Sai lech logits lon nhat (PyTorch vs ONNX): {max_diff:.6f}")
print(f"Ty le du doan nhan giong nhau: {giong_nhan * 100:.2f}%")
print("DAT" if (max_diff < 1e-3 and giong_nhan == 1.0) else "CHUA DAT - gui output cho giao vien")

# ---------- Do toc do voi 1 tin nhan (CPU) ----------
enc1 = ma_hoa([texts[0]])
for _ in range(10):
    chay_torch(enc1)
    chay_onnx(enc1)
N = 100
t0 = time.perf_counter()
for _ in range(N):
    chay_torch(enc1)
t_torch = (time.perf_counter() - t0) / N * 1000
t0 = time.perf_counter()
for _ in range(N):
    chay_onnx(enc1)
t_onnx = (time.perf_counter() - t0) / N * 1000
print(f"\nDo tre 1 tin tren CPU: PyTorch {t_torch:.1f} ms | ONNX {t_onnx:.1f} ms")