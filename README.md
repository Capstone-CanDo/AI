# LYNK AI Server

## Project Overview

LYNK AI Server는 QR 코드에 포함된 URL의 악성 여부를 분석하기 위한 FastAPI 기반 머신러닝 추론 서버입니다.

본 서버는 URL feature을 추출한 후 앙상블 모델을 이용하여 URL의 정상·악성 여부를 판별하며, SHAP(Shapley Additive Explanations)을 활용하여 모델의 예측 근거를 생성합니다.

또한 생성된 SHAP 값을 기반으로 사용자가 이해하기 쉬운 자연어 형태의 위험 설명을 제공하여 URL 분석 결과에 대한 해석 가능성을 높입니다.

---

## Tech Stack

### Backend
- Python 3.11
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- Random Forest
- SHAP

### Data Processing
- Pandas
- NumPy

### Deployment
- Docker

---

## Project Structure

```text
ai-server
├── .github              # GitHub Actions 및 저장소 설정
├── ensemble             # 앙상블 모델 관련 코드
├── models               # 학습된 모델 파일 저장
├── notebooks            # 모델 학습 및 실험용 노트북
├── src                  # URL 특징 추출 및 추론 로직
├── Dockerfile           # Docker 이미지 생성 설정
├── main.py              # FastAPI 서버 실행 파일
├── requirements.txt     # Python 패키지 의존성 목록
└── README.md
```

### Module Description

| Module | Description |
|----------|----------|
| ensemble | stacking 앙상블 모델 구성 |
| models | 학습 완료된 모델 파일 저장 |
| notebooks | 모델 학습, 평가 및 실험 코드 |
| src | URL 특징 추출, 예측 및 SHAP 설명 생성 |

---

## Prerequisites

다음 소프트웨어가 설치되어 있어야 합니다.

- Python 3.11 이상
- pip
- Git
- Docker (선택)

---

## Installation

### 1. Repository Clone

```bash
git clone https://github.com/Capstone-CanDo/ai-server.git
cd ai-server
```

### 2. Virtual Environment

```bash
python -m venv venv
```

#### Mac/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Server

FastAPI 서버를 실행합니다.

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

기본 실행 주소:

```text
http://127.0.0.1:8001
```

---

## Model Description

본 프로젝트는 URL 문자열 기반 특징을 활용하여 악성 URL을 탐지합니다.

추출된 특징은 다음과 같습니다.

- URL 길이
- Hostname 길이
- Path 길이
- Query 길이
- 특수문자 개수
- 숫자 개수
- 서브도메인 개수
- HTTPS 사용 여부
- 리다이렉션 정보

등 총 24개의 URL 특징을 사용합니다.

---

## Ensemble Model

본 프로젝트는 다음 세 가지 모델을 stacking 방식으로 결합한 앙상블 모델을 사용합니다.

- Random Forest
- XGBoost
- LightGBM

각 모델의 예측 확률을 평균하여 최종 악성 URL 여부를 판별합니다.

---

## SHAP-based Explanation

예측 결과에 대한 설명 가능성을 제공하기 위해 SHAP을 활용합니다.

모델이 위험하다고 판단한 주요 특징을 분석한 뒤 사용자에게 이해하기 쉬운 자연어 설명으로 변환하여 제공합니다.

예시:

- URL 길이가 비정상적으로 깁니다.
- 특수문자가 과도하게 포함되어 있습니다.
- 의심스러운 도메인 구조가 발견되었습니다.

---

## Open Source Libraries

본 프로젝트는 다음과 같은 오픈소스 라이브러리를 활용합니다.

- FastAPI
- Uvicorn
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Pandas
- NumPy
- Docker
