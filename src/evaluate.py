import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# --------------------------------------------------------
# 1. 평가에 사용할 데이터셋 로드
# --------------------------------------------------------

# ★ 반드시 baseline / finetuned 모두 같은 테스트셋을 사용해야 공정 비교가 됨
df_mal = pd.read_csv("data/processed/test/merged_phishing_test.csv", index_col=0)   # 악성
df_ben = pd.read_csv("data/processed/test/only_travel_normal.csv", index_col=0)  # 정상

df_ben['target'] = 0
df_mal['target'] = 1

df = pd.concat([df_mal, df_ben], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["url", "target"])
y = df["target"]

# --------------------------------------------------------
# 2. 모델 로드
# --------------------------------------------------------

baseline_model = joblib.load("models/baseline.pkl")
finetuned_model = joblib.load("models/finetuned.pkl")

print("📌 baseline.pkl & finetuned.pkl 로딩 완료\n")

# --------------------------------------------------------
# 3. 각 모델로 예측 수행
# --------------------------------------------------------

baseline_pred = baseline_model.predict(X)
finetune_pred = finetuned_model.predict(X)

# --------------------------------------------------------
# 4. 지표 계산 함수
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
# 5. baseline 모델 평가
# --------------------------------------------------------

evaluate_model("Baseline Model", y, baseline_pred)

# --------------------------------------------------------
# 6. finetuned 모델 평가
# --------------------------------------------------------

evaluate_model("Finetuned Model", y, finetune_pred)

# --------------------------------------------------------
# 7. 성능 비교 간단 요약
# --------------------------------------------------------

print("===============================================")
print("          Baseline vs Finetuned Summary        ")
print("===============================================")

def score_summary(y_true, pred):
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }

b = score_summary(y, baseline_pred)
f = score_summary(y, finetune_pred)

print("{:<12} {:>10} {:>10} {:>10} {:>10}".format("Model", "ACC", "PRE", "REC", "F1"))
print("-" * 56)
print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("Baseline", b["acc"], b["precision"], b["recall"], b["f1"]))
print("{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format("Finetuned", f["acc"], f["precision"], f["recall"], f["f1"]))
print()
