import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "sms_spam_collection.zip"

RAW_DIR.mkdir(parents=True, exist_ok=True)
print("Dang tai du lieu...")
urllib.request.urlretrieve(URL, ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH) as z:
    print("Cac tep trong zip:", z.namelist())
    z.extractall(RAW_DIR)
print("Xong. Kiem tra thu muc data/raw")
