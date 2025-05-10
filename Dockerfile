FROM python:3.9-slim

WORKDIR /app

# 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 기본 실행 명령어
ENTRYPOINT ["python", "src/api_server/app.py"]
