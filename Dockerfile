FROM python:3.11

# 작업 디렉토리
WORKDIR /app

# requirements 먼저 복사
COPY requirements.txt .

# 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# 서버 실행
CMD uvicorn main:app --host 0.0.0.0 --port $PORT