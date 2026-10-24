import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Chi dung train de hoc va val de danh gia. KHONG dung test o buoc nay.
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

X_train, y_train = train["text"], train["label"]
X_val, y_val = val["text"], val["label"]


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print("=" * 60)
    print(name)
    print("Ma tran nham lan (hang = that, cot = du doan, thu tu: ham, spam):")
    print(confusion_matrix(y_val, pred, labels=["ham", "spam"]))
    print(classification_report(
        y_val, pred, labels=["ham", "spam"], digits=3, zero_division=0))


# Co so 1: khong hoc gi, luon doan lop xuat hien nhieu nhat (ham)
evaluate("Co so 1: luon doan ham", DummyClassifier(strategy="most_frequent"))

# Co so 2: TF-IDF + Logistic Regression
# Pipeline dam bao TF-IDF chi hoc tren train, roi ap dung cho val
tfidf_lr = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
evaluate("Co so 2: TF-IDF + Logistic Regression", tfidf_lr)