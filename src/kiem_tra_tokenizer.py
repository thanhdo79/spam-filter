import pandas as pd
from tokenizers import Tokenizer
from transformers import AutoTokenizer

MAX_LEN = 64

# Bo tach tu nhe (se dung trong API)
nhe = Tokenizer.from_file("models/distilbert/tokenizer.json")
nhe.enable_truncation(max_length=MAX_LEN)
nhe.enable_padding(length=MAX_LEN, pad_id=nhe.token_to_id("[PAD]"),
                   pad_token="[PAD]")

# Bo tach tu day du cua transformers (dung luc huan luyen)
day_du = AutoTokenizer.from_pretrained("models/distilbert")

val = pd.read_csv("data/processed/val.csv")
khac = 0
for text in val["text"].astype(str):
    a = nhe.encode(text)
    b = day_du(text, truncation=True, max_length=MAX_LEN,
               padding="max_length")
    if a.ids != b["input_ids"] or a.attention_mask != b["attention_mask"]:
        khac += 1

print(f"So tin kiem tra: {len(val)} | So tin tach tu KHAC nhau: {khac}")
print("DAT" if khac == 0 else "CHUA DAT - gui output cho giao vien")