import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn

SEED = 42
MAX_LEN = 80      # so tu toi da moi tin (cat bot neu dai hon)
MIN_FREQ = 2      # tu xuat hien it hon 2 lan trong train -> <unk>
BATCH = 64
EPOCHS = 20
PATIENCE = 4      # dung som neu 4 epoch lien khong tien bo
EMB_DIM = 100
HIDDEN = 64
MODEL_NAME = "Bi-GRU (tu dien rieng)"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiet bi:", device)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", str(text).lower())


# Tu dien CHI xay tu tap train
counter = Counter(tok for t in train["text"] for tok in tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for tok, c in counter.items():
    if c >= MIN_FREQ:
        vocab[tok] = len(vocab)
print("Kich thuoc tu dien:", len(vocab))


def encode(text):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


def to_tensors(df):
    X = torch.tensor([encode(t) for t in df["text"]], dtype=torch.long)
    y = torch.tensor((df["label"] == "spam").astype(float).to_numpy(),
                     dtype=torch.float32)
    return X, y


class BiGRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.gru = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN * 2, 1)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1)
        out = out.masked_fill(~mask, -1e9)
        pooled = out.max(dim=1).values
        return self.fc(self.drop(pooled)).squeeze(-1)


Xtr, ytr = to_tensors(train)
Xva, yva = to_tensors(val)
y_true = yva.numpy().astype(int)

model = BiGRU(len(vocab)).to(device)
n_spam = float(ytr.sum())
n_ham = len(ytr) - n_spam
loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([n_ham / n_spam], device=device))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def predict(X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            logits = model(X[i:i + 256].to(device))
            preds.append((torch.sigmoid(logits) > 0.5).cpu())
    return torch.cat(preds).numpy().astype(int)


def prf(pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label=1, average="binary", zero_division=0)
    return p, r, f1


best_f1, best_state, wait = -1.0, None, 0
t0 = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    p, r, f1 = prf(predict(Xva))
    print(f"Epoch {epoch:2d} | loss {total / len(Xtr):.4f} | "
          f"val P {p:.3f} R {r:.3f} F1 {f1:.3f}")
    if f1 > best_f1:
        best_f1, wait = f1, 0
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Dung som (early stopping)")
            break
train_seconds = time.perf_counter() - t0

# Nap lai trong so tot nhat, danh gia lan cuoi
model.load_state_dict(best_state)
pred = predict(Xva)
p, r, f1 = prf(pred)
fp = int(((pred == 1) & (y_true == 0)).sum())
fn = int(((pred == 0) & (y_true == 1)).sum())
print(f"KET QUA TOT NHAT: P {p:.3f} R {r:.3f} F1 {f1:.3f} FP {fp} FN {fn}")

Path("models").mkdir(exist_ok=True)
torch.save({"state": best_state, "vocab": vocab}, "models/bigru.pt")

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