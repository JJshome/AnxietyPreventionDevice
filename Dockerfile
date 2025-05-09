FROM python:3.9-slim

LABEL maintainer="contact@example.com"
LABEL description="불안장애 예방장치 - 심박변이도 분석 및 두개전기자극 제어 시스템"

# 작업 디렉토리 설정
WORKDIR /app

# 블루투스 및 기타 필수 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluez \
    bluez-tools \
    libglib2.0-0 \
    libbluetooth-dev \
    pkg-config \
    libhdf5-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p data/ecg data/hrv data/models results

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 작동 포트 노출
EXPOSE 8080

# 실행 명령
CMD ["python", "src/api_server/app.py"]
