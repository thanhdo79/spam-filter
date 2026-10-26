# Anh nen: Python 3.13 ban gon, cung ban voi may phat trien
FROM python:3.13-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    THU_MUC_MO_HINH=/srv/models \
    DUONG_DAN_DB=/srv/data/app.db

# Cai thu vien truoc: Docker luu cache, sua ma nguon khong phai cai lai
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Chi sao chep ma nguon API (mo hinh duoc gan vao luc chay)
COPY app ./app

EXPOSE 8000

# Tu kiem tra suc khoe: goi /health moi 30 giay
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]