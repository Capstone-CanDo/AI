import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --------------------------------------------------------
# 1. 평가에 사용할 데이터셋 로드
# --------------------------------------------------------

df_mal = pd.read_csv("data/processed/test/merged_phishing_test.csv", index_col=0)   # 악성
df_ben = pd.read_csv("data/processed/test/only_travel_normal.csv", index_col=0)     # 정상

df_ben['target'] = 0
df_mal['target'] = 1

df = pd.concat([df_mal, df_ben], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["url", "target"])
y = df["target"]

# --------------------------------------------------------
# 2. 모델 로드 (Voting 앙상블 모델) 
# --------------------------------------------------------
model = joblib.load("models/ensemble_voting.pkl")

print("📌 ensemble_voting.pkl 로딩 완료\n")

# --------------------------------------------------------
# 3. 각 모델로 예측 수행
# --------------------------------------------------------

voting_pred = model.predict(X)

# --------------------------------------------------------
# 4. 지표 계산 함수
# --------------------------------------------------------

def score_summary(y_true, pred):
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

vote_score = score_summary(y, voting_pred)

# --------------------------------------------------------
# 5. 결과 출력
# --------------------------------------------------------

print("================================================")
print("        Ensemble Test Performance Summary       ")
print("================================================")

print("{:<12} {:>10} {:>10} {:>10} {:>10}".format("Model", "ACC", "PRE", "REC", "F1"))
print("-" * 56)

print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
    "Voting",
    vote_score["acc"],
    vote_score["precision"],
    vote_score["recall"],
    vote_score["f1"]
))

print()