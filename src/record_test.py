import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000   # 16 kHz, muc ma Whisper/Wav2Vec2 can
DURATION = 10         # so giay thu
OUTPUT = "data/raw/test.wav"

print("Danh sach thiet bi am thanh:")
print(sd.query_devices())

# Neu mic mac dinh khong phai Logitech, bo dau # o dong duoi
# va thay 1 bang so thu tu thiet bi Logitech trong danh sach tren:
# sd.default.device = (1, None)

input("Nhan Enter, roi noi tieng Viet trong 10 giay... ")
audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()  # cho thu xong

sf.write(OUTPUT, audio, SAMPLE_RATE)

peak = float(np.max(np.abs(audio)))
print(f"Da luu {OUTPUT}")
print(f"Do dai: {len(audio) / SAMPLE_RATE:.1f} giay")
print(f"Bien do dinh: {peak:.3f}")