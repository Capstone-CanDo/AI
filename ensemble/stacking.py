import pandas as pd
import joblib
import numpy as np
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------
# 테스트 데이터 로드 이 부분 확인해주세요!!!!
# --------------------------------------------------------
df_mal = pd.read_csv("data/processed/test/merged_phishing_test.csv", index_col=0)   # 악성
df_ben = pd.read_csv("data/processed/test/only_travel_normal.csv", index_col=0)     # 정상

df_ben["target"] = 0
df_mal["target"] = 1

df = pd.concat([df_mal, df_ben], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["url", "target"])
y = df["target"]

# --------------------------------------------------------
# 모델 로드 이 부분 맞게 수정해주세요!!
# --------------------------------------------------------
model1 = joblib.load("models/finetuned.pkl")
model2 = joblib.load("models/lightgbm_finetuned.pkl")
model3 = joblib.load("models/finetuned_xgboost.pkl")

print("📌 모델 3개 로딩 완료\n")

# --------------------------------------------------------
# 개별 모델 예측
# --------------------------------------------------------
pred1 = model1.predict(X)
pred2 = model2.predict(X)
pred3 = model3.predict(X)

# --------------------------------------------------------
# 확률 예측 (stacking feature 생성)
# --------------------------------------------------------
prob1 = model1.predict_proba(X)[:,1]
prob2 = model2.predict_proba(X)[:,1]
prob3 = model3.predict_proba(X)[:,1]

stack_X = np.column_stack([prob1, prob2, prob3])

# --------------------------------------------------------
# meta model 학습 (Stacking)
# --------------------------------------------------------
meta_model = LogisticRegression()

meta_model.fit(stack_X, y)

stack_pred = meta_model.predict(stack_X)

# --------------------------------------------------------
# Stacking 모델 저장
# --------------------------------------------------------
class StackingModel:

    def init(self, m1, m2, m3, meta):
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

#스태킹 모델 생성
stack_model = StackingModel(model1, model2, model3, meta_model)

os.makedirs("models", exist_ok=True)

save_path = "models/stacking_model.pkl"

joblib.dump(stack_model, save_path)

print(f"🎉 Stacking 모델 저장 완료 → {save_path}")