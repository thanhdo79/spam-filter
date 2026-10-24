import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42

# Chi dung train de hoc, val de so sanh. KHONG dung test o buoc nay.
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
X_train, y_train = train["text"], train["label"]
X_val = val["text"]
y_true = val["label"].to_numpy()


def make_pipeline(clf):
    # TF-IDF chi hoc tren train (nho Pipeline), tranh ro ri du lieu
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True)),
        ("clf", clf),
    ])


models = {
    "TF-IDF + LogReg": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=SEED)),
    "TF-IDF + SVM (tuyen tinh)": make_pipeline(
        LinearSVC(class_weight="balanced", random_state=SEED)),
    "TF-IDF + Random Forest": make_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED)),
}

rows = []
for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    pred = model.predict(X_val)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, pos_label="spam", average="binary", zero_division=0)
    fp = int(((pred == "spam") & (y_true == "ham")).sum())
    fn = int(((pred == "ham") & (y_true == "spam")).sum())

    rows.append({
        "model": name,
        "precision_spam": p,
        "recall_spam": r,
        "f1_spam": f1,
        "chan_nham_FP": fp,
        "bo_sot_FN": fn,
        "thoi_gian_hoc_giay": train_seconds,
    })

result = pd.DataFrame(rows).round(3)
print(result.to_string(index=False))

Path("reports").mkdir(exist_ok=True)
result.to_csv("reports/comparison_val.csv", index=False)
print("Da luu reports/comparison_val.csv")