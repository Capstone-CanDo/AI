# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os
import shap
import requests as req


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from utils.preprocess import extract_lexical_features, ensure_https

app = FastAPI()

# 서버 시작할 때 모델 한 번만 로드
model = joblib.load("models/finetuned.pkl")
explainer = shap.TreeExplainer(model) 

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
    shap_values: dict # feature별 기여도 
    redirect: dict # 리다이렉션 로그 정보 

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        url = ensure_https(request.url)
        redirect_info = get_redirect_chain(url)
        final_url = redirect_info["final_url"]

        feats = extract_lexical_features(url)
        feature_df = pd.DataFrame([feats])[FEATURE_COLS]

        prediction = model.predict(feature_df)[0]  # 0 or 1

        # SHAP 값 계산
        shap_vals = explainer.shap_values(feature_df)
        # 랜덤포레스트 이진분류: shap_values[1] = 피싱 클래스 기여도
        if isinstance(shap_vals, list):
            # 구버전: [정상 클래스, 피싱 클래스] 리스트
            phishing_shap = shap_vals[1][0].tolist()
        else:
            # 신버전: 3D array (n_samples, n_features, n_classes)
            phishing_shap = shap_vals[0, :, 1].tolist()

        contributions = dict(zip(FEATURE_COLS, phishing_shap))

        return PredictResponse(
            url=url,
            is_phishing=bool(prediction == 1),
            shap_values=contributions,
            redirect=redirect_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

def get_redirect_chain(url: str) -> dict:
    try:
        resp = req.get(
            url,
            allow_redirects=True,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},  # 봇 차단 우회
        )
        chain = [r.url for r in resp.history] + [resp.url]
        status_codes = [r.status_code for r in resp.history]

        return {
            "final_url": resp.url,
            "redirect_count": len(resp.history),
            "chain": chain,
            "status_codes": status_codes,
        }
    except req.exceptions.Timeout:
        return {"final_url": url, "redirect_count": 0, "chain": [url], "status_codes": [], "error": "timeout"}
    except req.exceptions.SSLError:
        return {"final_url": url, "redirect_count": 0, "chain": [url], "status_codes": [], "error": "ssl_error"}
    except Exception as e:
        return {"final_url": url, "redirect_count": 0, "chain": [url], "status_codes": [], "error": str(e)}