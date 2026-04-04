# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from utils.preprocess import extract_lexical_features, ensure_https

app = FastAPI()

# 서버 시작할 때 모델 한 번만 로드
model = joblib.load("models/finetuned.pkl")

# url feature 컬럼 순서 (학습 때랑 동일해야 함)
FEATURE_COLS = [
    "len_url", "len_hostname", "len_TLD", "len_path", "url_depth", "len_first_dir",
    "num_http", "num_https", "num_www", "num_@", "num_?", "num_&", "num_%",
    "num_#", "num_.", "num_=", "num__", "num_-", "num_hostname_-",
    "num_subdomains", "num_digits", "num_letters", "is_ip", "is_short_url"
]

class PredictRequest(BaseModel):
    url: str

class PredictResponse(BaseModel):
    url: str
    is_phishing: bool  # True = 악성, False = 정상

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        url = ensure_https(request.url)
        feats = extract_lexical_features(url)

        # url 컬럼 제외하고 feature만 추출
        feature_df = pd.DataFrame([feats])[FEATURE_COLS]

        prediction = model.predict(feature_df)[0]  # 0 or 1

        return PredictResponse(
            url=url,
            is_phishing=bool(prediction == 1),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}