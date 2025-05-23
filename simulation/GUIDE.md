# 불안장애 예방 시스템 시뮬레이션 가이드

본 문서는 특허 10-2022-0007209(불안장애 예방장치) 및 10-2459338(저주파 자극기 제어장치)를 기반으로 구현된 불안장애 예방 시스템의 시뮬레이션 환경 사용 방법을 안내합니다.

## 시뮬레이션 환경 개요

이 시뮬레이션 환경은 실제 하드웨어 없이도 불안장애 예방 시스템의 전체 워크플로우를 테스트하고 시연할 수 있도록 설계되었습니다.

시뮬레이션에서 제공하는 기능:
- ECG 신호 시뮬레이션 및 시각화
- HRV 분석 결과 실시간 표시
- 불안 수준 예측 및 시각화
- 저주파 자극 신호 시뮬레이션 및 시각화
- 자동 및 수동 자극 제어

## 시뮬레이션 시작하기

### 로컬 환경에서 실행

```bash
# 1. 필요한 패키지 설치
pip install -r requirements.txt

# 2. 시뮬레이션 실행
python simulation/run_simulation.py
```

### Docker를 사용한 실행

```bash
# 1. 시뮬레이션 Docker 이미지 빌드
docker build -t anxiety-prevention-simulation -f Dockerfile.simulation .

# 2. 시뮬레이션 실행 (GUI 모드)
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network host anxiety-prevention-simulation

# 3. 시뮬레이션 실행 (헤드리스 모드)
docker run -e HEADLESS=true anxiety-prevention-simulation --headless
```

### Docker Compose를 사용한 실행

```bash
# 개발 프로필로 실행 (시뮬레이션 포함)
docker-compose --profile dev up
```

## 시뮬레이션 사용법

### 기본 인터페이스

시뮬레이션 GUI는 다음과 같은 패널로 구성됩니다:

1. **ECG 모니터링 패널**: 시뮬레이션된 ECG 신호를 실시간으로 표시
2. **HRV 분석 패널**: 계산된 HRV 지표를 차트로 표시
3. **불안 모니터링 패널**: 예측된 불안 수준과 점수를 시각화
4. **자극 제어 패널**: 자극 파라미터 조정 및 자극 시작/중지
5. **시나리오 제어 패널**: 시뮬레이션 시나리오 선택 및 제어

### 시나리오 선택 및 실행

1. 시나리오 드롭다운 메뉴에서 시뮬레이션할 불안 패턴을 선택합니다:
   - 정상 (Normal): 정상적인 ECG 패턴
   - 경미한 불안 (Mild Anxiety): 경미한 불안 상태의 ECG 패턴
   - 중등도 불안 (Moderate Anxiety): 중등도 불안 상태의 ECG 패턴
   - 심각한 불안 (Severe Anxiety): 심각한 불안 상태의 ECG 패턴
   - 패닉 (Panic): 패닉 발작 상태의 ECG 패턴
   - 회복 (Recovery): 자극 후 회복 과정의 ECG 패턴
   - 커스텀 (Custom): 사용자 정의 패턴

2. "시작" 버튼을 클릭하여 선택한 시나리오를 시작합니다.

3. 시나리오 중에 실시간으로 변화하는 ECG 신호, HRV 지표, 불안 수준을 관찰합니다.

### 자극 제어

자동 자극 모드와 수동 자극 모드 중에서 선택할 수 있습니다:

#### 자동 자극
1. "자동 자극" 체크박스를 선택합니다.
2. 불안 임계값 슬라이더를 조정하여 자동 자극이 시작될 불안 점수를 설정합니다.
3. 시뮬레이션 중 불안 점수가 설정된 임계값을 초과하면 자동으로 자극이 시작됩니다.

#### 수동 자극
1. "자동 자극" 체크박스를 해제합니다.
2. 자극 파라미터를 조정합니다:
   - 주파수 (Hz): 0.5 ~ 100 Hz 범위에서 선택
   - 자극 강도 (mA): 0.5 ~ 4.0 mA 범위에서 선택
   - 지속 시간 (분): 5 ~ 30분 범위에서 선택
   - 파형 유형: 정현파, 구형파, 삼각파, 단상성, 양상성 등에서 선택
   - 위상 지연 (초): 0.1 ~ 1.0초 범위에서 선택

3. "자극 시작" 버튼을 클릭하여 수동으로 자극을 시작합니다.
4. "자극 중지" 버튼을 클릭하여 현재 진행 중인 자극을 중지합니다.

### 특허 기술 기반 스테레오 자극 시뮬레이션

이 시뮬레이션은 특허 10-2459338에 기반한 스테레오 자극 기능을 제공합니다:

1. **위상 지연 설정**:
   - "위상 지연" 슬라이더를 조정하여 두 자극기 간의 시간 지연을 설정합니다.
   - 지연 범위는 0.1초에서 1.0초까지 조정 가능합니다.

2. **자극 밸런스 설정**:
   - "좌/우 밸런스" 슬라이더를 조정하여 각 자극기의 상대적 강도를 설정합니다.
   - 중앙 위치는 두 자극기의 강도가 동일함을 의미합니다.
   - 슬라이더를 왼쪽 또는 오른쪽으로 이동하면 해당 측의 자극기 강도가 증가합니다.

3. **음악 기반 자극 (추가 기능)**:
   - "음악 기반 자극" 체크박스를 선택하면 음악 특성에 기반한 자극 패턴이 활성화됩니다.
   - 재생 버튼을 눌러 샘플 음악을 시작합니다.
   - 음악의 비트와 템포에 따라 자동으로 자극 패턴이 변경됩니다.

## 분석 및 결과 확인

### 실시간 HRV 분석

시뮬레이션은 다음과 같은 HRV 지표를 실시간으로 계산하고 표시합니다:

- **시간 영역 지표**:
  - SDNN (표준편차)
  - RMSSD (연속된 NN 간격 차이의 제곱 평균의 제곱근)
  - pNN50 (50ms보다 큰 NN 간격 차이의 비율)

- **주파수 영역 지표**:
  - LF (저주파 파워)
  - HF (고주파 파워)
  - LF/HF 비율

- **비선형 지표**:
  - SampEn (Sample Entropy)
  - SD1, SD2 (푸앵카레 플롯 지표)

### 불안 예측 결과

불안 예측 결과는 다음 정보를 제공합니다:

- **불안 수준**: 정상, 경미, 중등도, 심각, 패닉으로 분류
- **불안 점수**: 0-100 사이의 수치
- **발생 가능성**: 불안장애 발생 가능성(0-1)
- **불안 추세**: 증가, 감소, 안정

### 결과 저장

시뮬레이션 결과는 다음과 같은 방법으로 저장할 수 있습니다:

1. **데이터 저장**: "데이터 저장" 버튼을 클릭하여 현재까지의 ECG, HRV, 불안 예측 데이터를 저장합니다.
2. **보고서 생성**: "보고서 생성" 버튼을 클릭하여 시뮬레이션 세션의 요약 보고서를 생성합니다.
3. **스크린샷 저장**: "스크린샷" 버튼을 클릭하여 현재 화면의 이미지를 저장합니다.

저장된 데이터는 기본적으로 `./results` 디렉토리에 저장됩니다.

## 시뮬레이션 설정 사용자 정의

`simulation/simulation_config.json` 파일을 편집하여 시뮬레이션 설정을 사용자 정의할 수 있습니다:

```json
{
  "simulation": {
    "duration": 1800,
    "real_time_factor": 1.0,
    "headless": false
  },
  "ecg_simulator": {
    "sample_rate": 250,
    "noise_level": 0.03,
    "baseline_hr": 70
  },
  ...
}
```

## 헤드리스 모드

헤드리스 모드는 GUI 없이 시뮬레이션을 실행하는 방식으로, 자동화된 테스트나 서버 환경에서 유용합니다:

```bash
python simulation/run_simulation.py --headless
```

헤드리스 모드에서는 로그 파일 및 CSV 파일로 결과가 저장됩니다.

## 시뮬레이션 API

시뮬레이션은 다른 애플리케이션과의 통합을 위한 간단한 HTTP API를 제공합니다. API는 기본적으로 http://localhost:8081에서 사용할 수 있습니다:

- `GET /status` - 현재 시뮬레이션 상태 반환
- `POST /scenario/{scenario_name}` - 시나리오 시작
- `POST /stimulation/start` - 자극 시작
- `POST /stimulation/stop` - 자극 중지
- `GET /data` - 현재 시뮬레이션 데이터 반환

API 사용 예시:
```bash
# 시뮬레이션 상태 확인
curl http://localhost:8081/status

# 중등도 불안 시나리오 시작
curl -X POST http://localhost:8081/scenario/moderate_anxiety

# 자극 시작
curl -X POST -H "Content-Type: application/json" -d '{"frequency": 30, "intensity": 2.0, "duration": 1200}' http://localhost:8081/stimulation/start
```

## 시뮬레이션과 실제 시스템의 차이점

이 시뮬레이션은 교육 및 개발 목적으로 설계되었으며, 실제 의료 기기와는 다음과 같은 차이가 있습니다:

1. 시뮬레이션된 ECG 신호는 실제 생체 신호의 복잡성을 완전히 반영하지 않습니다.
2. 불안 예측 알고리즘은 실제 의료적 검증 없이 특허 내용을 기반으로 구현되었습니다.
3. 자극 효과는 단순화된 모델을 사용하여 시뮬레이션됩니다.

실제 의료 목적으로는 적절한 인증을 받은 의료 기기를 사용해야 합니다.

## 문제 해결

시뮬레이션 실행 중 발생할 수 있는 일반적인 문제와 해결 방법:

1. **GUI 표시 문제**:
   - Docker에서 실행 시 `--network host` 옵션을 사용했는지 확인하세요.
   - X11 연결 권한을 설정하세요: `xhost +local:docker`

2. **성능 문제**:
   - 실시간 요소가 중요하지 않은 경우 `real_time_factor`를 높여 시뮬레이션 속도를 높일 수 있습니다.
   - 리소스 사용량이 높은 경우 `sample_rate`를 낮추거나 `plot_history`를 줄이세요.

3. **오류 메시지**:
   - 로그 파일은 `./logs` 디렉토리에 저장됩니다.
   - 상세 디버그 로그를 활성화하려면 `--log-level debug` 옵션을 사용하세요.
