import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib


df_mal = pd.read_csv("data/baseline/pre-train/phishing_output.csv", index_col=0)   # 악성만
df_ben = pd.read_csv("data/baseline/pre-train/normal_1000_preprocessed.csv", index_col=0)      # 정상만

#라벨, 악성 1, 정상 0
df_mal['target'] = 1
df_ben['target'] = 0

df = pd.concat([df_mal, df_ben], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

drop_cols = ["url"]

X = df.drop(columns=drop_cols + ["target"])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# 랜덤 포레스트 모델 학습 
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:")
print(classification_report(y_test, y_pred))


# Feature Importance 출력

importances = model.feature_importances_
feature_names = X.columns

print("\n Feature Importances:")
for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20} : {score:.4f}")


# 모델 자동 저장 
os.makedirs("models", exist_ok=True)   # 폴더 없으면 생성
save_path = "models/baseline.pkl"

joblib.dump(model, save_path)

print(f"\n🎉 모델 저장 완료 → {save_path}")
