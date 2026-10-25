import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

THU_MUC_MO_HINH = Path(os.getenv("THU_MUC_MO_HINH", "models"))
DUONG_DAN_ONNX = THU_MUC_MO_HINH / "distilbert.onnx"
DUONG_DAN_TOKENIZER = THU_MUC_MO_HINH / "distilbert" / "tokenizer.json"
MAX_LEN = 64
NGUONG_SPAM = 0.5  # xac suat spam >= nguong thi ket luan la spam

# Luu mo hinh da nap de moi yeu cau dung lai
TRANG_THAI = {}


def tao_tokenizer(duong_dan):
    tok = Tokenizer.from_file(str(duong_dan))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding(
        length=MAX_LEN,
        pad_id=tok.token_to_id("[PAD]"),
        pad_token="[PAD]",
    )
    return tok


@asynccontextmanager
async def lifespan(app: FastAPI):
    TRANG_THAI["tokenizer"] = tao_tokenizer(DUONG_DAN_TOKENIZER)
    TRANG_THAI["session"] = ort.InferenceSession(
        str(DUONG_DAN_ONNX), providers=["CPUExecutionProvider"])
    yield
    TRANG_THAI.clear()


app = FastAPI(title="Smart Anti-Spam API", lifespan=lifespan)


class YeuCau(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class KetQua(BaseModel):
    label: str
    spam_probability: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "session" in TRANG_THAI}


@app.post("/predict", response_model=KetQua)
def predict(yeu_cau: YeuCau):
    t0 = time.perf_counter()
    enc = TRANG_THAI["tokenizer"].encode(yeu_cau.text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    logits = TRANG_THAI["session"].run(
        None, {"input_ids": ids, "attention_mask": mask})[0][0]

    # softmax on dinh: tru gia tri lon nhat de tranh tran so
    e = np.exp(logits - logits.max())
    p_spam = float(e[1] / e.sum())

    return KetQua(
        label="spam" if p_spam >= NGUONG_SPAM else "ham",
        spam_probability=round(p_spam, 4),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )