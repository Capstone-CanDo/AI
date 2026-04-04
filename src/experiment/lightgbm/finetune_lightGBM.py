import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

# ------------------------------------------------------------
# 1. baseline 데이터 로드
# ------------------------------------------------------------
df_base_normal = pd.read_csv("data/processed/baseline/normal_2.csv", index_col=0)
df_base_phish  = pd.read_csv("data/processed/baseline/phishing_2.csv", index_col=0)

df_base_normal["target"] = 0
df_base_phish["target"] = 1

# ------------------------------------------------------------
# 2. 파인튜닝 데이터 로드
# ------------------------------------------------------------
df_ft_normal = pd.read_csv("data/processed/fine-tuning/finetuning_data_normal.csv", index_col=0)
df_ft_phish  = pd.read_csv("data/processed/fine-tuning/finetuning_data_phishing.csv", index_col=0)

df_ft_normal["target"] = 0
df_ft_phish["target"] = 1

# ------------------------------------------------------------
# 3. baseline + finetune 데이터 합치기
# ------------------------------------------------------------
df_all = pd.concat([
    df_base_normal, df_base_phish,
    df_ft_normal, df_ft_phish
], ignore_index=True)

# 데이터 셔플
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------------------------
# 4. Feature / Label 분리
# ------------------------------------------------------------
drop_cols = ["url"]

X = df_all.drop(columns=drop_cols + ["target"])
y = df_all["target"]

# ------------------------------------------------------------
# 5. Train/Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# ------------------------------------------------------------
# 6. LightGBM 모델 학습
# ------------------------------------------------------------
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. 성능 평가
# ------------------------------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 8. Feature Importance 출력
# ------------------------------------------------------------
importances = model.feature_importances_
feature_names = X.columns

print("\nFeature Importances:")
for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20} : {score:.4f}")

# ------------------------------------------------------------
# 9. 모델 저장
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)
save_path = "models/lightgbm_finetuned.pkl"

joblib.dump(model, save_path)
print(f"\n🎉 모델 저장 완료 → {save_path}")