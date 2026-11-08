import random

from locust import HttpUser, between, task

# Mot so tin mau de gui thu (khong phai du lieu that cua nguoi dung)
TIN_SPAM = [
    "Free entry in a weekly competition, text WIN to 80086 now",
    "Congratulations! You have won a 1000 pound prize. Call 09061701461 now",
    "URGENT! Your mobile number has been selected for a cash award, claim today",
    "Get a free ringtone, reply YES to 87077 to subscribe",
]
TIN_HAM = [
    "Ok, see you at lunch tomorrow",
    "Can you send me the notes from the lecture today?",
    "I will be home late, do not wait for dinner",
    "Happy birthday! Hope you have a great day",
]


class NguoiDung(HttpUser):
    # Moi nguoi dung ao nghi ngau nhien 0.5 den 1.5 giay giua cac lan
    wait_time = between(0.5, 1.5)

    @task(8)
    def du_doan(self):
        """Viec chinh: gui mot tin de loc (ti le 8 phan)."""
        text = random.choice(TIN_SPAM if random.random() < 0.4 else TIN_HAM)
        self.client.post("/predict", json={"text": text}, name="/predict")

    @task(1)
    def xem_thong_ke(self):
        self.client.get("/stats", name="/stats")

    @task(1)
    def xem_gan_day(self):
        self.client.get("/recent?limit=20", name="/recent")