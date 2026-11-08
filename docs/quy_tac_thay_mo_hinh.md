# Quy tac thay mo hinh phat hien spam

## 1. Muc dich
Quy dinh khi nao mot mo hinh ung vien duoc phep thay mo hinh dang chay, va thay nhu the nao cho an toan.
Quy tac duoc khai bao truoc khi chay thi nghiem va khong duoc sua sau khi da thay ket qua.

## 2. Quy tac (khai bao truoc khi chay buoc 6c-3b)
Ung vien chi duoc de cu thay mo hinh khi dong thoi dat ca ba dieu kien:

| Dieu kien | Noi dung | Du lieu do |
|---|---|---|
| (a) | Bat duoc it nhat 5/11 tin spam tieng Viet | Bo do co dinh 50 tin |
| (b) | F1 spam tren UCI test khong giam qua 0.01 so voi mo hinh hien tai (0.952) | UCI test |
| (c) | So ham bi chan nham (FP) tren bo do khong tang qua 2 so voi mo hinh hien tai (2), tuc toi da 4 | Bo do co dinh 50 tin |

Nguyen tac di kem:
- Bo do co dinh 50 tin va UCI test chi do mot lan cho moi ung vien. Khong dung de chon cau hinh.
- Chon epoch theo F1 tren UCI val + 30 tin tieng Viet val.
- Neu muon thu cai tien tiep thi phai co bo do moi va khai bao quy tac moi truoc khi chay.
- Khong noi long dieu kien sau khi da thay ket qua.
- Dieu kien (c) uu tien han che chan nham vi chan nham tin hop le (ma OTP, thong bao ngan hang) ton kem hon bo sot.

## 3. Quy trinh thay mo hinh an toan (chi ap dung khi du ca ba dieu kien)
1. Sao luu file mo hinh ONNX dang chay (models/distilbert.onnx) thanh ban dat ten co ngay.
2. Xuat ung vien ra ONNX (chinh src/xuat_onnx.py tro toi models/distilbert_vi).
3. Chay src/kiem_tra_tokenizer.py de chac chan tokenizer khop voi mo hinh.
4. Chay pytest va CI phai xanh.
5. Build lai Docker, goi thu /predict voi cac tin mau.
6. Theo doi Prometheus (do tre, ti le spam du doan) mot thoi gian truoc khi coi la on dinh.
7. Dang ky phien ban moi trong MLflow Model Registry, dat alias champion cho ban moi va giu ban cu de quay lai.
8. Cach quay lai: tra file ONNX da sao luu, khoi dong lai container, dat lai alias champion.

## 4. Nhat ky quyet dinh

### Quyet dinh lan 1 - 08/11/2026: KHONG THAY
- Ung vien: DistilBERT huan luyen lai voi UCI train + 90 tin tieng Viet (du lieu do tro ly soan, nhom ra soat).
- Run MLflow: experiment huan_luyen_lai_phan_hoi (id 2), run ae26b81c7d484ece894c2bd114922974.
- Ma nguon: commit f7901f3 (MLflow ghi 893f17b vi script chua duoc commit luc chay).

| Chi so | Mo hinh hien tai | Ung vien |
|---|---|---|
| Spam tieng Viet bat duoc (bo do 50 tin) | 0/11 | 11/11 |
| Spam tieng Anh bat duoc (bo do 50 tin) | 10/11 | 11/11 |
| FP tren bo do 50 tin | 2 | 9 |
| FN tren bo do 50 tin | 12 | 0 |
| F1 spam tren bo do 50 tin | 0.588 | 0.83 |
| F1 spam tren UCI test | 0.952 | 0.945 |

Doi chieu quy tac: (a) DAT, (b) DAT, (c) KHONG DAT (FP 9 > 4).
Ket luan: khong thay mo hinh dang chay. Ung vien duoc giu o models/distilbert_vi de lam challenger.
Luu y: F1 tong the tren bo do cua ung vien cao hon, nhung quyet dinh theo quy tac da khai bao truoc (uu tien han che chan nham).

## 5. Gioi han
- Bo do chi co 22 tin spam va 28 tin ham; moi tin sai thay doi chi so nhieu.
- Du lieu train va do do cung mot nguon soan, co the cung mot thien lech.
- Chi chay mot cau hinh va mot hat giong, chua biet muc dao dong giua cac lan huan luyen.
- Nguyen nhan FP tang chua duoc kiem chung; gia thuyet: du lieu bo sung co ti le spam 40% cao hon UCI (khoang 13%).