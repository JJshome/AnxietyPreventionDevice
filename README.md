# 불안장애 예방장치 (Anxiety Prevention Device)

이 프로젝트는 웨어러블 심전도 센서와 저주파 자극기(TENS)를 활용한 불안장애 예방 시스템의 구현체입니다. 심박변이도(HRV) 분석을 통해 불안장애 발생 가능성을 예측하고, 필요시 두개전기자극을 통해 불안장애를 예방합니다.

<p align="center">
  <img src="docs/images/logo.svg" alt="불안장애 예방장치 로고" width="200" />
</p>

## 시스템 개요

불안장애 예방 시스템은 아래 다이어그램과 같이 동작합니다:

<p align="center">
  <img src="docs/images/system-overview.svg" alt="시스템 개요" width="800" />
</p>

## 기술 개요

1. **심박변이도(HRV) 모니터링**: 웨어러블 ECG 센서를 통해 실시간으로 사용자의 심전도를 측정하고 HRV를 분석합니다.
2. **불안 예측 알고리즘**: HRV 패턴 분석을 통해 불안장애 발생 가능성을 예측합니다.
3. **저주파 전기자극**: 불안장애 발생 가능성이 높을 때 두 개의 자극기를 통해 위상차를 가진 저주파 전기자극을 제공합니다.
4. **무선 제어 시스템**: 블루투스를 통해 자극기를 무선으로 제어합니다.

## HRV 분석 과정

심박변이도(HRV) 분석은 불안장애 예측의 핵심 기술입니다:

<p align="center">
  <img src="docs/images/hrv-analysis.svg" alt="HRV 분석 과정" width="800" />
</p>

## 저주파 자극기 스테레오 제어

특허 기술(10-2459338)에 기반한 저주파 자극기 스테레오 제어 시스템:

<p align="center">
  <img src="docs/images/stimulation-modes.svg" alt="자극기 스테레오 제어" width="800" />
</p>

## 프로젝트 구조

```
AnxietyPreventionDevice/
├── src/                 # 소스 코드
│   ├── hrv_analyzer.py      # HRV 분석 모듈
│   ├── anxiety_predictor.py # 불안 예측 모듈
│   ├── stim_controller.py   # 자극기 제어 모듈
│   └── anxiety_prevention_controller.py  # 전체 시스템 제어
├── simulation/         # 시뮬레이션 환경
│   ├── ecg_simulator.py     # ECG 신호 시뮬레이터
│   └── stimulator_simulator.py  # 자극기 시뮬레이터
├── app/                # 애플리케이션
│   ├── cli/                # 명령줄 인터페이스
│   └── web/                # 웹 인터페이스
├── utils/              # 유틸리티 함수
├── docs/               # 문서
│   └── images/             # 이미지 및 다이어그램
└── data/               # 데이터 저장소
```

## 시스템 워크플로우

1. 웨어러블 심전도 센서로부터 ECG 신호를 수신합니다.
2. 수신된 ECG 신호에서 HRV 특성(SDNN, RMSSD, pNN50, LF/HF 비율 등)을 추출합니다.
3. 추출된 HRV 특성을 기반으로 불안장애 발생 가능성을 예측합니다.
4. 불안장애 발생 가능성이 임계값을 초과하면 저주파 자극기 활성화 신호를 전송합니다.
5. 블루투스를 통해 연결된 두 개의 저주파 자극기에 서로 다른 위상의 자극 신호를 전송합니다.
6. 자극 결과를 모니터링하고 필요시 자극 패턴을 조정합니다.

자세한 워크플로우는 [워크플로우 문서](docs/workflow.md)를 참조하세요.

## 설치 및 실행

### 요구사항
- Python 3.8+
- numpy, scipy, pandas
- matplotlib, seaborn
- Flask (웹 인터페이스용)

### 설치
```bash
pip install -r requirements.txt
```

### 시뮬레이션 실행
```bash
python -m simulation.run
```

### 웹 인터페이스 실행
```bash
python -m app.web.app
```

## 라이선스

이 프로젝트는 특허 기술을 기반으로 하고 있습니다:
- 불안장애 예방장치 (특허 출원번호: 10-2020-0085095)
- 저주파 자극기 제어장치 (등록특허 10-2459338)
