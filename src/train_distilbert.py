import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
CKPT = "distilbert-base-uncased"   # mo hinh tien huan luyen
MAX_LEN = 64
BATCH = 16
EPOCHS = 3
LR = 2e-5                          # learning rate nho khi tinh chinh
MODEL_NAME = "DistilBERT (tinh chinh)"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiet bi:", device)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

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
ids_va, mask_va, yva = to_tensors(val)
y_true = yva.numpy()

# num_labels=2: them lop phan loai moi (ham / spam), lop nay chua duoc hoc
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


def prf(pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label=1, average="binary", zero_division=0)
    return p, r, f1


best_f1, best_state = -1.0, None
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
    p, r, f1 = prf(predict(ids_va, mask_va))
    print(f"Epoch {epoch} | loss {total / len(ids_tr):.4f} | "
          f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    if f1 > best_f1:
        best_f1 = f1
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
train_seconds = time.perf_counter() - t0

# Nap lai epoch tot nhat va danh gia lan cuoi tren val
model.load_state_dict(best_state)
pred = predict(ids_va, mask_va)
p, r, f1 = prf(pred)
fp = int(((pred == 1) & (y_true == 0)).sum())
fn = int(((pred == 0) & (y_true == 1)).sum())
print(f"KET QUA TOT NHAT: P {p:.3f} R {r:.3f} F1 {f1:.3f} FP {fp} FN {fn}")

# Luu mo hinh (thu muc models/ da bi .gitignore, khong day len GitHub)
save_dir = Path("models/distilbert")
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

row = {
    "model": MODEL_NAME,
    "precision_spam": p,
    "recall_spam": r,
    "f1_spam": f1,
    "chan_nham_FP": fp,
    "bo_sot_FN": fn,
    "thoi_gian_hoc_giay": train_seconds,
}
out_path = Path("reports/comparison_val.csv")
new = pd.DataFrame([row]).round(3)
if out_path.exists():
    old = pd.read_csv(out_path)
    old = old[old["model"] != MODEL_NAME]
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(out_path, index=False)
print(new.to_string(index=False))