import pandas as pd
import joblib
import numpy as np
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------
# 1. 테스트 데이터 로드 이 부분 확인해주세요!!!!
# --------------------------------------------------------

df_mal = pd.read_csv("../data/test/phishing_features_250_output.csv", index_col=0)
df_ben = pd.read_csv("../data/test/balanced_250_features.csv", index_col=0)

df_ben["target"] = 0
df_mal["target"] = 1

df = pd.concat([df_mal, df_ben], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.columns.tolist())
X = df.drop(columns=["target"])
y = df["target"]

# --------------------------------------------------------
# 2. 모델 로드 이 부분 맞게 수정해주세요!!
# --------------------------------------------------------

model1 = joblib.load("models/finetuned.pkl")
model2 = joblib.load("models/finetuned_xgboost.pkl")
model3 = joblib.load("models/lightgbm_finetuned.pkl")

print("📌 모델 3개 로딩 완료\n")

# --------------------------------------------------------
# 3. 개별 모델 예측
# --------------------------------------------------------

pred1 = model1.predict(X)
pred2 = model2.predict(X)
pred3 = model3.predict(X)

# --------------------------------------------------------
# 4. 확률 예측 (stacking feature 생성)
# --------------------------------------------------------

prob1 = model1.predict_proba(X)[:,1]
prob2 = model2.predict_proba(X)[:,1]
prob3 = model3.predict_proba(X)[:,1]

stack_X = np.column_stack([prob1, prob2, prob3])

# --------------------------------------------------------
# 5. meta model 학습 (Stacking)
# --------------------------------------------------------

meta_model = LogisticRegression()

meta_model.fit(stack_X, y)

stack_pred = meta_model.predict(stack_X)

# --------------------------------------------------------
# 6. 평가 함수
# --------------------------------------------------------

def evaluate_model(name, y_true, y_pred):

    print(f"==================== {name} ====================")

    print(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score   : {f1_score(y_true, y_pred):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n")

# --------------------------------------------------------
# 7. 모델 평가
# --------------------------------------------------------


evaluate_model("Stacking Model", y, stack_pred)

# --------------------------------------------------------
# 8. 성능 비교 표
# --------------------------------------------------------

print("===============================================")
print("           Model Performance Summary           ")
print("===============================================")

def score_summary(y_true, pred):
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }

s1 = score_summary(y, pred1)
s2 = score_summary(y, pred2)
s3 = score_summary(y, pred3)
ss = score_summary(y, stack_pred)

print("{:<12} {:>10} {:>10} {:>10} {:>10}".format("Model", "ACC", "PRE", "REC", "F1"))
print("-"*60)

print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("Randomforest", s1["acc"], s1["precision"], s1["recall"], s1["f1"]))
print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("XGBoost", s2["acc"], s2["precision"], s2["recall"], s2["f1"]))
print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("lightGBM", s3["acc"], s3["precision"], s3["recall"], s3["f1"]))
print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("Stacking", ss["acc"], ss["precision"], ss["recall"], ss["f1"]))

print()

# --------------------------------------------------------
# 9. Stacking 모델 저장
# --------------------------------------------------------

class StackingModel:

    def __init__(self, m1, m2, m3, meta):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.meta = meta

    def predict(self, X):

        p1 = self.m1.predict_proba(X)[:,1]
        p2 = self.m2.predict_proba(X)[:,1]
        p3 = self.m3.predict_proba(X)[:,1]

        stack_X = np.column_stack([p1, p2, p3])

        return self.meta.predict(stack_X)

# 스태킹 모델 생성
stack_model = StackingModel(model1, model2, model3, meta_model)

os.makedirs("models", exist_ok=True)

save_path = "models/stacking_model.pkl"

joblib.dump(stack_model, save_path)

print(f"🎉 Stacking 모델 저장 완료 → {save_path}")