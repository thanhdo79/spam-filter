from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main


class GiaTokenizer:
    """Bo tach tu gia: danh dau ids[1] = 1 neu tin co chu 'free'."""

    def encode(self, text):
        ids = [0] * main.MAX_LEN
        ids[1] = 1 if "free" in text.lower() else 0
        mask = [1] * 3 + [0] * (main.MAX_LEN - 3)
        return SimpleNamespace(ids=ids, attention_mask=mask)


class GiaPhienChay:
    """Phien ONNX gia: tra logits 'spam' neu ids[1] == 1, nguoc lai 'ham'."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, _ten_dau_ra, dau_vao):
        la_spam = dau_vao["input_ids"][0][1] == 1
        logits = [-3.0, 3.0] if la_spam else [3.0, -3.0]
        return [np.array([logits], dtype=np.float32)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Moi bai thu dung co so du lieu tam rieng, khong dung vao data/app.db
    monkeypatch.setattr(main, "DUONG_DAN_DB", tmp_path / "test.db")
    monkeypatch.setattr(main, "tao_tokenizer", lambda _duong_dan: GiaTokenizer())
    monkeypatch.setattr(main.ort, "InferenceSession", GiaPhienChay)
    with TestClient(main.app) as c:
        yield c


TIN_SPAM = "Free entry in a weekly competition, text WIN to 80086 now"
TIN_HAM = "Ok, see you at lunch tomorrow"


def gui_du_doan(client, text):
    return client.post("/predict", json={"text": text})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_tin_spam(client):
    r = gui_du_doan(client, TIN_SPAM)
    assert r.status_code == 200
    kq = r.json()
    assert kq["label"] == "spam"
    assert kq["spam_probability"] > 0.9
    assert kq["id"] == 1
    assert kq["latency_ms"] >= 0


def test_predict_tin_ham(client):
    kq = gui_du_doan(client, TIN_HAM).json()
    assert kq["label"] == "ham"
    assert kq["spam_probability"] < 0.1


def test_predict_tin_rong_bi_tu_choi(client):
    assert gui_du_doan(client, "").status_code == 422


def test_predict_thieu_truong_bi_tu_choi(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_tin_qua_dai_bi_tu_choi(client):
    assert gui_du_doan(client, "a" * 2001).status_code == 422


def test_feedback_duoc_ghi_nhan(client):
    id_tin = gui_du_doan(client, TIN_SPAM).json()["id"]
    r = client.post("/feedback", json={"id": id_tin, "correct_label": "ham"})
    assert r.status_code == 200
    s = client.get("/stats").json()
    assert s["so_phan_hoi"] == 1
    assert s["so_phan_hoi_cho_thay_sai"] == 1  # mo hinh doan spam, nhan dung la ham


def test_feedback_id_khong_ton_tai(client):
    r = client.post("/feedback", json={"id": 9999, "correct_label": "ham"})
    assert r.status_code == 404


def test_feedback_nhan_khong_hop_le(client):
    id_tin = gui_du_doan(client, TIN_SPAM).json()["id"]
    r = client.post("/feedback", json={"id": id_tin, "correct_label": "abc"})
    assert r.status_code == 422


def test_stats_khi_chua_co_du_lieu(client):
    s = client.get("/stats").json()
    assert s["tong"] == 0
    assert s["so_spam"] == 0
    assert s["ty_le_spam"] == 0.0


def test_stats_sau_nhieu_tin(client):
    gui_du_doan(client, TIN_SPAM)
    gui_du_doan(client, TIN_SPAM)
    gui_du_doan(client, TIN_HAM)
    s = client.get("/stats").json()
    assert s["tong"] == 3
    assert s["so_spam"] == 2
    assert s["so_ham"] == 1
    assert s["ty_le_spam"] == pytest.approx(0.6667, abs=1e-4)


def test_recent_tra_tin_moi_nhat_truoc(client):
    gui_du_doan(client, TIN_SPAM)
    gui_du_doan(client, TIN_HAM)
    ds = client.get("/recent").json()
    assert [b["id"] for b in ds] == [2, 1]
    assert len(client.get("/recent?limit=1").json()) == 1


def test_recent_limit_khong_hop_le(client):
    assert client.get("/recent?limit=0").status_code == 422


def test_admin_tra_ve_trang_html(client):
    r = client.get("/admin")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Smart Anti-Spam" in r.text


def test_admin_khong_chen_html_nguy_hiem(client):
    # Bao ve chong loi XSS: trang khong duoc chen noi dung bang cac API chen HTML
    # (tim cach DUNG that, co dau cham, nen khong bat nham chu thich)
    trang = client.get("/admin").text
    cam = [".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write"]
    for cach in cam:
        assert cach not in trang, f"Trang dang dung {cach}"