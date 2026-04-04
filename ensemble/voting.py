import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# ------------------------------------------------------------
# 4. Feature / Label 분리
# ------------------------------------------------------------
drop_cols = ["url"]

X = df_all.drop(columns=drop_cols + ["target"])
y = df_all["target"]

# ------------------------------------------------------------
# 5. Train/Test Split (동일한 test 비율 사용)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

#랜덤포레스트 모델 학습
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

#XGB 모델 학습
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss"
)

#lightGBM 모델 학습 
lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    n_jobs=-1,
    random_state=42
)

#Voting 앙상블 
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    voting='soft'
)

voting_model.fit(X_train, y_train)

joblib.dump(voting_model, "models/ensemble_voting.pkl")
print("Voting 앙상블 모델 저장 → models/ensemble_voting.pkl")

