import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from phishing_feature import ensure_https, extract_lexical_features


df_mal = pd.read_csv("phishing_output.csv", index_col=0)   # 악성만
df_ben = pd.read_csv("normal_1000_preprocessed.csv", index_col=0)      # 정상만

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

# -----------------------------------------
# 7. Feature Importance 출력
# -----------------------------------------
import numpy as np

importances = model.feature_importances_
feature_names = X.columns

print("\n Feature Importances:")
for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:20} : {score:.4f}")



# =============================
# 2. 새로운 URL 예측 기능
# =============================

def predict_new_url(url):
    """새로운 URL 하나를 입력하면 악성인지 정상인지 예측하는 함수"""

    # URL을 https로 변환
    url = ensure_https(url)

    # 피처 추출
    feats = extract_lexical_features(url)

    # DataFrame 포맷 맞추기
    df_new = pd.DataFrame([feats])

    # 학습한 feature 컬럼 순서 맞추기
    df_new = df_new[X.columns]

    # 모델 예측
    pred = model.predict(df_new)[0]

    return "악성 URL" if pred == 1 else "정상 URL"


# =============================
# 3. 직접 실행 시 예측 모드
# =============================
if __name__ == "__main__":
    while True:
        url = input("\n확인할 URL 입력 (종료: exit): ")

        if url.lower() == "exit":
            break

        result = predict_new_url(url)
        print("결과:", result)