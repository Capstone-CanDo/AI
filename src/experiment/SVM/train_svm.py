import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

# ------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------

df_mal = pd.read_csv("../data/baseline/phishing_2.csv", index_col=0)   # 악성
df_ben = pd.read_csv("../data/baseline/normal_2.csv", index_col=0)     # 정상

# 라벨 설정 (악성 1 / 정상 0)
df_mal["target"] = 1
df_ben["target"] = 0

df = pd.concat([df_mal, df_ben], ignore_index=True)

# 데이터 셔플
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------------------------
# 2. Feature / Label 분리
# ------------------------------------------------------------

drop_cols = ["url"]

X = df.drop(columns=drop_cols + ["target"])
y = df["target"]

# ------------------------------------------------------------
# 3. Train / Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# ------------------------------------------------------------
# 4. SVM 모델 (Scaling 포함)
# ------------------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. 성능 평가
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 6. 모델 저장
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)
save_path = "models/baseline_svm.pkl"

joblib.dump(model, save_path)

print(f"\n🎉 모델 저장 완료 → {save_path}")