import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

THU_MUC_MO_HINH = Path(os.getenv("THU_MUC_MO_HINH", "models"))
DUONG_DAN_ONNX = THU_MUC_MO_HINH / "distilbert.onnx"
DUONG_DAN_TOKENIZER = THU_MUC_MO_HINH / "distilbert" / "tokenizer.json"
DUONG_DAN_DB = Path(os.getenv("DUONG_DAN_DB", "data/app.db"))
DUONG_DAN_ADMIN = Path(__file__).parent / "static" / "admin.html"
MAX_LEN = 64
NGUONG_SPAM = 0.5  # xac suat spam >= nguong thi ket luan la spam

# Luu mo hinh da nap de moi yeu cau dung lai
TRANG_THAI = {}

# ---------- So lieu giam sat (Prometheus) ----------
SO_DU_DOAN = Counter(
    "spam_predictions_total", "So tin da phan loai, chia theo nhan", ["label"])
THOI_GIAN_SUY_LUAN = Histogram(
    "spam_inference_seconds", "Thoi gian tach tu + suy luan ONNX (giay)",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0))
SO_PHAN_HOI = Counter(
    "spam_feedback_total",
    "So phan hoi cua nguoi dung, chia theo mo hinh dung hay sai", ["ket_qua"])
SO_YEU_CAU_HTTP = Counter(
    "spam_http_requests_total", "So yeu cau HTTP",
    ["duong_dan", "phuong_thuc", "ma_trang_thai"])
THOI_GIAN_HTTP = Histogram(
    "spam_http_request_seconds", "Thoi gian xu ly yeu cau HTTP (giay)",
    ["duong_dan"])


def tao_tokenizer(duong_dan):
    tok = Tokenizer.from_file(str(duong_dan))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding(
        length=MAX_LEN,
        pad_id=tok.token_to_id("[PAD]"),
        pad_token="[PAD]",
    )
    return tok


def ket_noi():
    DUONG_DAN_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DUONG_DAN_DB)
    conn.row_factory = sqlite3.Row
    return conn


def khoi_tao_db():
    with closing(ket_noi()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS du_doan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thoi_gian TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                latency_ms REAL NOT NULL,
                nhan_dung TEXT,
                thoi_gian_phan_hoi TEXT
            )
            """
        )
        conn.commit()


def bay_gio():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    khoi_tao_db()
    TRANG_THAI["tokenizer"] = tao_tokenizer(DUONG_DAN_TOKENIZER)
    TRANG_THAI["session"] = ort.InferenceSession(
        str(DUONG_DAN_ONNX), providers=["CPUExecutionProvider"])
    yield
    TRANG_THAI.clear()


app = FastAPI(title="Smart Anti-Spam API", lifespan=lifespan)


@app.middleware("http")
async def do_yeu_cau_http(request: Request, call_next):
    """Do moi yeu cau HTTP: dem so luong va do thoi gian xu ly."""
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    # Dung mau duong dan (vd /predict) thay vi dia chi that de so lieu khong phinh ra
    duong_dan = route.path if route else "khac"
    if duong_dan != "/metrics":
        SO_YEU_CAU_HTTP.labels(
            duong_dan, request.method, str(response.status_code)).inc()
        THOI_GIAN_HTTP.labels(duong_dan).observe(time.perf_counter() - t0)
    return response


class YeuCau(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class KetQua(BaseModel):
    id: int
    label: str
    spam_probability: float
    latency_ms: float


class PhanHoi(BaseModel):
    id: int
    correct_label: Literal["ham", "spam"]


class ThongKe(BaseModel):
    tong: int
    so_spam: int
    so_ham: int
    ty_le_spam: float
    latency_tb_ms: float
    so_phan_hoi: int
    so_phan_hoi_cho_thay_sai: int


class BanGhi(BaseModel):
    id: int
    thoi_gian: str
    text: str
    label: str
    spam_probability: float
    nhan_dung: str | None = None


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
    nhan = "spam" if p_spam >= NGUONG_SPAM else "ham"
    # do tre chi tinh phan tach tu + suy luan, khong tinh ghi co so du lieu
    giay_suy_luan = time.perf_counter() - t0
    THOI_GIAN_SUY_LUAN.observe(giay_suy_luan)
    do_tre = round(giay_suy_luan * 1000, 2)

    with closing(ket_noi()) as conn:
        cur = conn.execute(
            "INSERT INTO du_doan "
            "(thoi_gian, text, label, spam_probability, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (bay_gio(), yeu_cau.text, nhan, round(p_spam, 4), do_tre),
        )
        conn.commit()
        id_moi = cur.lastrowid

    SO_DU_DOAN.labels(nhan).inc()
    return KetQua(id=id_moi, label=nhan,
                  spam_probability=round(p_spam, 4), latency_ms=do_tre)


@app.post("/feedback")
def feedback(phan_hoi: PhanHoi):
    with closing(ket_noi()) as conn:
        hang = conn.execute(
            "SELECT label FROM du_doan WHERE id = ?", (phan_hoi.id,)
        ).fetchone()
        if hang is None:
            raise HTTPException(status_code=404, detail="Khong tim thay id")
        conn.execute(
            "UPDATE du_doan SET nhan_dung = ?, thoi_gian_phan_hoi = ? "
            "WHERE id = ?",
            (phan_hoi.correct_label, bay_gio(), phan_hoi.id),
        )
        conn.commit()
    ket_qua = "dung" if hang["label"] == phan_hoi.correct_label else "sai"
    SO_PHAN_HOI.labels(ket_qua).inc()
    return {"status": "da_ghi_nhan"}


@app.get("/stats", response_model=ThongKe)
def stats():
    with closing(ket_noi()) as conn:
        r = conn.execute(
            """
            SELECT COUNT(*) AS tong,
                   COALESCE(SUM(label = 'spam'), 0) AS so_spam,
                   COALESCE(AVG(latency_ms), 0) AS tb_latency,
                   COALESCE(SUM(nhan_dung IS NOT NULL), 0) AS so_phan_hoi,
                   COALESCE(SUM(nhan_dung IS NOT NULL
                                AND nhan_dung != label), 0) AS so_sai
            FROM du_doan
            """
        ).fetchone()
    tong = int(r["tong"])
    so_spam = int(r["so_spam"])
    return ThongKe(
        tong=tong,
        so_spam=so_spam,
        so_ham=tong - so_spam,
        ty_le_spam=round(so_spam / tong, 4) if tong else 0.0,
        latency_tb_ms=round(float(r["tb_latency"]), 2),
        so_phan_hoi=int(r["so_phan_hoi"]),
        so_phan_hoi_cho_thay_sai=int(r["so_sai"]),
    )


@app.get("/recent", response_model=list[BanGhi])
def recent(limit: int = Query(20, ge=1, le=200)):
    with closing(ket_noi()) as conn:
        rows = conn.execute(
            "SELECT id, thoi_gian, text, label, spam_probability, nhan_dung "
            "FROM du_doan ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [BanGhi(**dict(r)) for r in rows]


@app.get("/admin", include_in_schema=False)
def admin():
    return FileResponse(DUONG_DAN_ADMIN)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus goi dia chi nay moi 15 giay de lay so lieu."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)