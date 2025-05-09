"""
불안장애 예방 시스템 중앙 제어기

심전도 센서 데이터 수집, HRV 분석, 불안 예측, 자극 제어를 통합하는 중앙 제어 모듈입니다.
전체 워크플로우를 조율하고 각 컴포넌트 간의 통신을 관리합니다.
"""

import logging
import time
import threading
import asyncio
import queue
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
import json
from enum import Enum
from datetime import datetime
import uuid

# 시스템 구성요소 임포트
from src.ecg_sensor.ecg_interface import ECGInterface
from src.ecg_sensor.data_processor import ECGDataProcessor
from src.hrv_analyzer.hrv_analysis import HRVAnalyzer
from src.hrv_analyzer.hrv_anxiety_predictor import HRVAnxietyPredictor
from src.stimulation_controller.stereo_stimulator import StereoStimulator, WaveformType, PhaseType

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SystemState(Enum):
    """시스템 상태 열거형"""
    IDLE = "idle"             # 대기 중
    MONITORING = "monitoring" # 모니터링 중
    ANALYZING = "analyzing"   # 분석 중
    STIMULATING = "stimulating"  # 자극 중
    ERROR = "error"           # 오류 상태


class OperationMode(Enum):
    """작동 모드 열거형"""
    AUTOMATIC = "automatic"   # 자동 모드 (분석 결과에 따라 자동 자극)
    MANUAL = "manual"         # 수동 모드 (사용자 명령에 따라 자극)
    PREVENTION = "prevention" # 예방 모드 (낮은 임계값으로 조기 자극)
    TRAINING = "training"     # 훈련 모드 (정기적인 자극 제공)


class AnxietyLevel(Enum):
    """불안 수준 열거형"""
    NORMAL = 0        # 정상
    MILD = 1          # 경도 불안
    MODERATE = 2      # 중등도 불안
    HIGH = 3          # 고도 불안
    SEVERE = 4        # 심각한 불안


class AnxietyPreventionController:
    """
    불안장애 예방장치 중앙 제어 컨트롤러
    
    불안장애 예방장치(특허 10-2022-0007209)와 저주파 자극기 제어장치(특허 10-2459338)를
    통합하여 심전도 모니터링, HRV 분석, 불안 예측, 그리고 저주파 자극을 제어합니다.
    """
    
    def __init__(self, config_file=None):
        """
        AnxietyPreventionController 초기화
        
        매개변수:
            config_file (str, optional): 설정 파일 경로
        """
        # 기본 설정
        self.config = {
            'ecg': {
                'sampling_rate': 256,  # Hz
                'window_size': 300,    # 초
                'buffer_size': 300 * 256,  # sampling_rate * window_size
            },
            'hrv': {
                'control_sampen_mean': 1.4,
                'anxiety_threshold': 0.75,
                'analysis_interval': 60,  # 초
            },
            'stimulation': {
                'num_stimulators': 2,
                'base_frequency': 10,  # Hz
                'waveform_type': 'sine',
                'phase_type': 'biphasic',
                'phase_delay': 0.5,    # 초
                'amplitude': 0.5,
                'duration': 180,       # 초
                'cooldown': 300,       # 자극 후 휴지기 (초)
            },
            'operation': {
                'mode': 'automatic',
                'bluetooth_enabled': True,
                'data_recording': True,
                'data_directory': 'data',
            }
        }
        
        # 설정 파일 로드
        if config_file:
            self._load_config(config_file)
        
        # 작동 상태 변수
        self.state = SystemState.IDLE
        self.mode = OperationMode(self.config['operation']['mode'])
        self.running = False
        self.monitoring_thread = None
        self.analysis_thread = None
        self.event_queue = queue.Queue()
        self.session_id = None
        
        # 데이터 버퍼
        self.ecg_buffer = []
        self.rri_buffer = []
        self.hrv_results = []
        self.anxiety_scores = []
        self.timestamps = []
        self.last_stimulation_time = 0
        
        # 컴포넌트 초기화
        self._init_components()
        
        logger.info("불안장애 예방 컨트롤러가 초기화되었습니다.")
    
    def _load_config(self, config_file):
        """
        설정 파일을 로드합니다.
        
        매개변수:
            config_file (str): 설정 파일 경로
        """
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            # 설정 업데이트
            for section in loaded_config:
                if section in self.config:
                    self.config[section].update(loaded_config[section])
            
            logger.info(f"설정 파일 '{config_file}'가 로드되었습니다.")
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            logger.info("기본 설정을 사용합니다.")
    
    def _init_components(self):
        """
        시스템 구성 요소들을 초기화합니다.
        """
        try:
            # ECG 센서 인터페이스 초기화
            self.ecg_sensor = ECGInterface(
                sampling_rate=self.config['ecg']['sampling_rate'],
                buffer_size=self.config['ecg']['buffer_size'],
                bluetooth_enabled=self.config['operation']['bluetooth_enabled']
            )
            
            # HRV 분석 및 불안 예측 모듈 초기화
            self.hrv_analyzer = HRVAnxietyPredictor(
                sampling_rate=self.config['ecg']['sampling_rate'],
                control_sampen_mean=self.config['hrv']['control_sampen_mean'],
                anxiety_threshold=self.config['hrv']['anxiety_threshold'],
                window_size=self.config['ecg']['window_size']
            )
            
            # 스테레오 자극기 컨트롤러 초기화
            self.stimulator = StereoStimulator(
                num_stimulators=self.config['stimulation']['num_stimulators'],
                base_frequency=self.config['stimulation']['base_frequency'],
                waveform_type=WaveformType(self.config['stimulation']['waveform_type']),
                phase_type=PhaseType(self.config['stimulation']['phase_type']),
                phase_delay=self.config['stimulation']['phase_delay'],
                amplitude=self.config['stimulation']['amplitude']
            )
            
            logger.info("모든 구성 요소가 초기화되었습니다.")
        except Exception as e:
            logger.error(f"구성 요소 초기화 중 오류 발생: {e}")
            self.state = SystemState.ERROR
    
    def start(self):
        """
        불안장애 예방 시스템을 시작합니다.
        """
        if self.running:
            logger.warning("시스템이 이미 실행 중입니다.")
            return
        
        self.running = True
        self.session_id = str(uuid.uuid4())
        
        # 데이터 버퍼 초기화
        self.ecg_buffer = []
        self.rri_buffer = []
        self.hrv_results = []
        self.anxiety_scores = []
        self.timestamps = []
        
        # ECG 센서 시작
        try:
            self.ecg_sensor.connect()
            logger.info("ECG 센서가 연결되었습니다.")
        except Exception as e:
            logger.error(f"ECG 센서 연결 실패: {e}")
            self.state = SystemState.ERROR
            self.running = False
            return
        
        # 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # 분석 스레드 시작
        self.analysis_thread = threading.Thread(
            target=self._analysis_worker
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self.state = SystemState.MONITORING
        logger.info(f"불안장애 예방 시스템이 시작되었습니다. 세션 ID: {self.session_id}")
        
        return self.session_id
    
    def stop(self):
        """
        불안장애 예방 시스템을 중지합니다.
        """
        if not self.running:
            logger.warning("시스템이 실행 중이지 않습니다.")
            return
        
        self.running = False
        
        # 자극 중지
        if self.state == SystemState.STIMULATING:
            self.stimulator.stop_stimulation()
        
        # ECG 센서 중지
        try:
            self.ecg_sensor.disconnect()
            logger.info("ECG 센서가 연결 해제되었습니다.")
        except Exception as e:
            logger.error(f"ECG 센서 연결 해제 중 오류 발생: {e}")
        
        # 스레드 종료 대기
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # 결과 저장
        self._save_session_results()
        
        self.state = SystemState.IDLE
        logger.info(f"불안장애 예방 시스템이 중지되었습니다. 세션 ID: {self.session_id}")
    
    def _monitoring_worker(self):
        """
        심전도 모니터링 워커 함수
        """
        try:
            logger.info("ECG 모니터링이 시작되었습니다.")
            
            while self.running:
                # ECG 데이터 수집
                ecg_data = self.ecg_sensor.read_data()
                
                if ecg_data is not None and len(ecg_data) > 0:
                    # ECG 데이터 처리
                    self._process_ecg_data(ecg_data)
                
                # 잠시 대기
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"ECG 모니터링 중 오류 발생: {e}")
            self.state = SystemState.ERROR
            self.event_queue.put({"type": "error", "message": str(e)})
    
    def _process_ecg_data(self, ecg_data):
        """
        수집된 ECG 데이터를 처리합니다.
        
        매개변수:
            ecg_data (ndarray): 수집된 ECG 데이터
        """
        try:
            # ECG 버퍼에 데이터 추가
            self.ecg_buffer.extend(ecg_data)
            
            # 버퍼 크기 제한
            max_buffer_size = self.config['ecg']['buffer_size']
            if len(self.ecg_buffer) > max_buffer_size:
                self.ecg_buffer = self.ecg_buffer[-max_buffer_size:]
            
            # R-피크 탐지
            if len(self.ecg_buffer) >= self.config['ecg']['sampling_rate'] * 10:  # 최소 10초 데이터
                # 최근 추가된 데이터에 대해서만 R-피크 탐지
                ecg_segment = np.array(self.ecg_buffer)
                
                try:
                    # R-피크 탐지 (neurokit2 등 외부 라이브러리 적용 가능)
                    # 여기서는 간단한 구현만 제공
                    # 실제 구현에서는 더 정교한 알고리즘 사용 필요
                    peaks, _ = self._detect_r_peaks(ecg_segment)
                    
                    # R-R 간격 계산
                    rri = np.diff(peaks) / self.config['ecg']['sampling_rate'] * 1000  # ms 단위
                    
                    # RRI 버퍼에 추가
                    if len(rri) > 0:
                        self.rri_buffer.extend(rri)
                        
                        # 이벤트 큐에 RRI 데이터 추가
                        self.event_queue.put({
                            "type": "rri_update",
                            "data": rri.tolist(),
                            "timestamp": time.time()
                        })
                except Exception as e:
                    logger.warning(f"R-피크 탐지 중 오류 발생: {e}")
        
        except Exception as e:
            logger.error(f"ECG 데이터 처리 중 오류 발생: {e}")
    
    def _detect_r_peaks(self, ecg_data):
        """
        ECG 데이터에서 R-피크를 탐지합니다.
        
        매개변수:
            ecg_data (ndarray): ECG 데이터
            
        반환값:
            tuple: (R-피크 위치, R-피크 값)
        """
        # 샘플링 레이트
        fs = self.config['ecg']['sampling_rate']
        
        # 간단한 R-피크 탐지 알고리즘 (실제 구현에서는 더 정교한 알고리즘 적용 필요)
        # 데이터를 필터링하여 기준선 노이즈 제거
        filtered = self._filter_ecg(ecg_data)
        
        # 미분
        diff = np.diff(filtered)
        squared = diff**2
        
        # 이동 평균 필터링
        window_size = int(0.1 * fs)  # 100ms 윈도우
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 임계값 적용
        threshold = 0.5 * np.max(integrated)
        
        # 피크 검출
        peaks = []
        for i in range(1, len(integrated)-1):
            if integrated[i-1] < integrated[i] and integrated[i] > integrated[i+1] and integrated[i] > threshold:
                peaks.append(i)
        
        # 최소 거리 적용 (250ms)
        min_distance = int(0.25 * fs)
        filtered_peaks = []
        
        if peaks:
            filtered_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peak)
        
        return np.array(filtered_peaks), integrated[filtered_peaks] if filtered_peaks else np.array([])
    
    def _filter_ecg(self, ecg_data):
        """
        ECG 데이터를 필터링합니다.
        
        매개변수:
            ecg_data (ndarray): ECG 데이터
            
        반환값:
            ndarray: 필터링된 ECG 데이터
        """
        # 샘플링 레이트
        fs = self.config['ecg']['sampling_rate']
        
        # 주파수 대역 설정
        low_freq = 5.0   # Hz, 대역통과 필터의 하한
        high_freq = 15.0  # Hz, 대역통과 필터의 상한
        
        # 필터 설계
        nyquist = fs / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 대역통과 필터 적용
        from scipy import signal
        b, a = signal.butter(2, [low, high], 'bandpass')
        filtered = signal.filtfilt(b, a, ecg_data)
        
        return filtered
    
    def _analysis_worker(self):
        """
        HRV 분석 및 불안 예측 워커 함수
        """
        analysis_interval = self.config['hrv']['analysis_interval']  # 초
        last_analysis_time = 0
        
        try:
            logger.info("HRV 분석이 시작되었습니다.")
            
            while self.running:
                # 이벤트 큐에서 이벤트 처리
                self._process_events()
                
                current_time = time.time()
                
                # 분석 간격마다 HRV 분석 수행
                if current_time - last_analysis_time >= analysis_interval and len(self.rri_buffer) > 30:
                    self.state = SystemState.ANALYZING
                    
                    # RRI 데이터로 HRV 분석 및 불안 예측
                    rri_data = np.array(self.rri_buffer[-500:])  # 최대 500개의 최근 RRI 데이터
                    timestamp = current_time
                    
                    analysis_result = self.hrv_analyzer.predict_anxiety(rri_data, timestamp)
                    
                    # 분석 결과 처리
                    self._handle_analysis_result(analysis_result, timestamp)
                    
                    last_analysis_time = current_time
                    self.state = SystemState.MONITORING
                
                # 잠시 대기
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"HRV 분석 중 오류 발생: {e}")
            self.state = SystemState.ERROR
            self.event_queue.put({"type": "error", "message": str(e)})
    
    def _process_events(self):
        """
        이벤트 큐의 이벤트를 처리합니다.
        """
        # 큐의 모든 이벤트 처리
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                
                if event["type"] == "error":
                    logger.error(f"이벤트 에러: {event['message']}")
                    
                elif event["type"] == "stimulation_complete":
                    logger.info(f"자극 완료: {event['session_id']}")
                    self.state = SystemState.MONITORING
                    
                elif event["type"] == "manual_stimulation":
                    # 수동 자극 요청 처리
                    self._start_stimulation(
                        event.get("duration", self.config['stimulation']['duration']),
                        event.get("parameters", {})
                    )
                    
                # 다른 이벤트 타입은 필요에 따라 추가
                
            except Exception as e:
                logger.error(f"이벤트 처리 중 오류 발생: {e}")
    
    def _handle_analysis_result(self, analysis_result, timestamp):
        """
        HRV 분석 및 불안 예측 결과를 처리합니다.
        
        매개변수:
            analysis_result (dict): 분석 결과
            timestamp (float): 타임스탬프
        """
        try:
            # 결과 저장
            self.hrv_results.append(analysis_result)
            self.anxiety_scores.append(analysis_result['anxiety_score'])
            self.timestamps.append(timestamp)
            
            # 로그 기록
            anxiety_score = analysis_result['anxiety_score']
            prediction = analysis_result['prediction']
            logger.info(f"불안 점수: {anxiety_score:.2f}, 예측: {prediction}")
            
            # 자동 모드에서 자극 필요 여부 결정
            if self.mode == OperationMode.AUTOMATIC:
                if analysis_result['stimulation_recommended']:
                    # 마지막 자극 후 충분한 시간이 지났는지 확인
                    cooldown_time = self.config['stimulation']['cooldown']
                    if timestamp - self.last_stimulation_time >= cooldown_time:
                        logger.info(f"불안 점수 {anxiety_score:.2f}로 자극이 권장됩니다.")
                        self._start_stimulation()
                    else:
                        logger.info(f"불안 점수 {anxiety_score:.2f}이지만 쿨다운 중입니다. " +
                                   f"다음 자극까지 {cooldown_time - (timestamp - self.last_stimulation_time):.0f}초")
            
            # 예방 모드에서 더 낮은 임계값으로 자극
            elif self.mode == OperationMode.PREVENTION:
                prevention_threshold = self.config['hrv']['anxiety_threshold'] * 0.8  # 20% 낮은 임계값
                if anxiety_score >= prevention_threshold:
                    if timestamp - self.last_stimulation_time >= self.config['stimulation']['cooldown']:
                        logger.info(f"예방 모드: 불안 점수 {anxiety_score:.2f}로 자극이 시작됩니다.")
                        self._start_stimulation()
            
            # 훈련 모드는 여기서 처리하지 않음 (별도의 스케줄링 필요)
            
        except Exception as e:
            logger.error(f"분석 결과 처리 중 오류 발생: {e}")
    
    def _start_stimulation(self, duration=None, parameters=None):
        """
        저주파 자극을 시작합니다.
        
        매개변수:
            duration (int, optional): 자극 지속 시간 (초)
            parameters (dict, optional): 자극 매개변수
        """
        if self.state == SystemState.STIMULATING:
            logger.warning("자극이 이미 진행 중입니다.")
            return
        
        # 지속 시간 설정
        if duration is None:
            duration = self.config['stimulation']['duration']
        
        # 파라미터 설정
        stim_params = {}
        if parameters:
            stim_params.update(parameters)
        
        # 자극 시작
        self.stimulator.start_stimulation(duration=duration)
        self.state = SystemState.STIMULATING
        self.last_stimulation_time = time.time()
        
        logger.info(f"자극이 시작되었습니다. 지속 시간: {duration}초")
        
        # 자극 완료 콜백 설정 (별도 스레드에서 실행)
        def completion_callback():
            time.sleep(duration)
            if self.state == SystemState.STIMULATING:
                self.state = SystemState.MONITORING
                logger.info("자극이 완료되었습니다.")
                self.event_queue.put({
                    "type": "stimulation_complete",
                    "session_id": self.session_id,
                    "timestamp": time.time()
                })
        
        completion_thread = threading.Thread(target=completion_callback)
        completion_thread.daemon = True
        completion_thread.start()
    
    def stop_stimulation(self):
        """
        진행 중인 자극을 중지합니다.
        """
        if self.state != SystemState.STIMULATING:
            logger.warning("자극이 진행 중이지 않습니다.")
            return
        
        self.stimulator.stop_stimulation()
        self.state = SystemState.MONITORING
        logger.info("자극이 중지되었습니다.")
    
    def set_operation_mode(self, mode):
        """
        시스템 작동 모드를 설정합니다.
        
        매개변수:
            mode (str): 작동 모드 ('automatic', 'manual', 'prevention', 'training')
        """
        try:
            self.mode = OperationMode(mode)
            logger.info(f"작동 모드가 {mode}로 설정되었습니다.")
        except ValueError:
            logger.error(f"유효하지 않은 작동 모드: {mode}")
    
    def manual_stimulation(self, duration=None, parameters=None):
        """
        수동으로 자극을 시작합니다.
        
        매개변수:
            duration (int, optional): 자극 지속 시간 (초)
            parameters (dict, optional): 자극 매개변수
        """
        if not self.running:
            logger.warning("시스템이 실행 중이지 않습니다.")
            return
        
        if duration is None:
            duration = self.config['stimulation']['duration']
        
        # 이벤트 큐에 수동 자극 요청 추가
        self.event_queue.put({
            "type": "manual_stimulation",
            "duration": duration,
            "parameters": parameters,
            "timestamp": time.time()
        })
        
        logger.info(f"수동 자극 요청이 추가되었습니다. 지속 시간: {duration}초")
    
    def set_stimulation_parameters(self, parameters):
        """
        자극 파라미터를 설정합니다.
        
        매개변수:
            parameters (dict): 자극 파라미터
        """
        if 'waveform_type' in parameters:
            self.stimulator.set_waveform(parameters['waveform_type'])
        
        if 'phase_type' in parameters:
            self.stimulator.set_waveform(self.stimulator.waveform_type, parameters['phase_type'])
        
        if 'frequency' in parameters:
            self.stimulator.set_frequency(parameters['frequency'])
        
        if 'amplitude' in parameters:
            self.stimulator.set_amplitude(parameters['amplitude'])
        
        if 'phase_delay' in parameters:
            self.stimulator.set_phase_delay(parameters['phase_delay'])
        
        # 설정 업데이트
        for key, value in parameters.items():
            if key in self.config['stimulation']:
                self.config['stimulation'][key] = value
        
        logger.info(f"자극 파라미터가 업데이트되었습니다: {parameters}")
    
    def _save_session_results(self):
        """
        세션 결과를 저장합니다.
        """
        if not self.config['operation']['data_recording']:
            return
        
        try:
            import os
            
            # 데이터 디렉토리 생성
            data_dir = self.config['operation']['data_directory']
            os.makedirs(data_dir, exist_ok=True)
            
            # 세션 정보
            session_data = {
                "session_id": self.session_id,
                "start_time": self.timestamps[0] if self.timestamps else time.time(),
                "end_time": time.time(),
                "config": self.config,
                "anxiety_scores": self.anxiety_scores,
                "timestamps": self.timestamps,
                "hrv_results": self.hrv_results
            }
            
            # 저장
            filename = f"{data_dir}/session_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(session_data, f, default=str)
            
            logger.info(f"세션 결과가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"세션 결과 저장 중 오류 발생: {e}")
    
    def get_status(self):
        """
        현재 시스템 상태 정보를 반환합니다.
        
        반환값:
            dict: 상태 정보
        """
        return {
            "state": self.state.value,
            "mode": self.mode.value,
            "running": self.running,
            "session_id": self.session_id,
            "ecg_buffer_size": len(self.ecg_buffer),
            "rri_buffer_size": len(self.rri_buffer),
            "anxiety_score": self.anxiety_scores[-1] if self.anxiety_scores else None,
            "last_analysis_time": self.timestamps[-1] if self.timestamps else None,
            "last_stimulation_time": self.last_stimulation_time,
            "stimulator_status": {
                "phase_delay": self.stimulator.phase_delay,
                "waveform_type": self.stimulator.waveform_type.value,
                "phase_type": self.stimulator.phase_type.value,
                "base_frequency": self.stimulator.base_frequency,
                "amplitude": self.stimulator.amplitude,
                "running": self.stimulator.running
            }
        }


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 컨트롤러 생성
    controller = AnxietyPreventionController()
    
    print("불안장애 예방 시스템 - 중앙 제어 컨트롤러")
    print("시스템을 시작하려면 start() 메서드를 호출하세요.")
