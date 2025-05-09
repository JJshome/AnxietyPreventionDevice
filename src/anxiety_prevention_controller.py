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

# 시스템 구성요소 임포트
from ecg_sensor.sensor_interface import ECGSensorInterface
from ecg_sensor.data_processor import ECGDataProcessor
from hrv_analyzer.hrv_analysis import HRVAnalyzer
from hrv_analyzer.hrv_anxiety_predictor import HRVAnxietyPredictor, AnxietyLevel
from stimulation_controller.stereo_stimulator import StereoStimulator, WaveformType

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SystemState(Enum):
    """시스템 상태 열거형"""
    INITIALIZED = "initialized"
    CONNECTING = "connecting"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    STIMULATING = "stimulating"
    ERROR = "error"
    STOPPED = "stopped"


class AnxietyPreventionController:
    """
    불안장애 예방 시스템 중앙 제어기
    
    전체 시스템의 워크플로우를 관리하고 각 컴포넌트 간의 데이터 흐름을 조율합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        중앙 제어기 초기화
        
        Args:
            config_path: 설정 파일 경로 (None인 경우 기본 설정 사용)
        """
        # 시스템 상태 초기화
        self.state = SystemState.INITIALIZED
        self.running = False
        self.error_message = ""
        
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 구성요소 초기화
        self.ecg_sensor = None
        self.ecg_processor = ECGDataProcessor()
        self.hrv_analyzer = HRVAnalyzer()
        self.anxiety_predictor = HRVAnxietyPredictor(
            model_path=self.config.get("anxiety_model_path")
        )
        self.stimulator = StereoStimulator()
        
        # 데이터 큐 및 버퍼
        self.ecg_data_queue = queue.Queue(maxsize=1000)
        self.ecg_buffer = []
        self.hrv_buffer = []
        
        # 작업 스레드
        self.data_processing_thread = None
        self.analysis_thread = None
        
        # 콜백 및 이벤트 리스너
        self.event_listeners = {
            "state_change": [],
            "ecg_data": [],
            "hrv_result": [],
            "anxiety_prediction": [],
            "stimulation_start": [],
            "stimulation_stop": [],
            "error": []
        }
        
        # 기타 설정
        self.analysis_interval = self.config.get("analysis_interval", 60)  # 초
        self.last_analysis_time = 0
        self.auto_stimulation = self.config.get("auto_stimulation", True)
        self.anxiety_threshold = self.config.get("anxiety_threshold", 70)  # 자동 자극 임계값
        
        logger.info("AnxietyPreventionController initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            설정 딕셔너리
        """
        default_config = {
            "analysis_interval": 60,  # 초 단위 HRV 분석 간격
            "ecg_buffer_size": 300,  # 초 단위 ECG 버퍼 크기
            "anxiety_threshold": 70,  # 자동 자극 임계값 (0-100)
            "auto_stimulation": True,  # 자동 자극 활성화 여부
            "stimulation_default_params": {
                "frequency": 30.0,  # Hz
                "pulse_width": 100.0,  # us
                "amplitude": 2.0,  # mA
                "waveform": "biphasic",
                "duration": 1200,  # 초 (20분)
                "phase_delay": 0.5,  # 초
            },
            "log_level": "INFO",
            "data_save_path": "./data/",
            "anxiety_model_path": None,
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # 사용자 설정으로 기본 설정 업데이트
                    default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.info("Using default configuration")
                
        return default_config
        
    def register_ecg_sensor(self, sensor: ECGSensorInterface) -> bool:
        """
        심전도 센서 등록
        
        Args:
            sensor: 심전도 센서 인터페이스 구현 객체
            
        Returns:
            성공 여부
        """
        if self.running:
            logger.warning("Cannot register sensor while system is running")
            return False
            
        self.ecg_sensor = sensor
        logger.info(f"ECG sensor registered: {sensor}")
        return True
        
    def register_stimulator(self, stimulator_id: str, stimulator_interface) -> bool:
        """
        자극기 등록
        
        Args:
            stimulator_id: 자극기 식별자
            stimulator_interface: 자극기 인터페이스 구현 객체
            
        Returns:
            성공 여부
        """
        if self.running:
            logger.warning("Cannot register stimulator while system is running")
            return False
            
        result = self.stimulator.register_stimulator(stimulator_id, stimulator_interface)
        if result:
            logger.info(f"Stimulator registered: {stimulator_id}")
        return result
        
    def add_event_listener(self, event_type: str, callback: Callable) -> bool:
        """
        이벤트 리스너 등록
        
        Args:
            event_type: 이벤트 유형
            callback: 콜백 함수
            
        Returns:
            성공 여부
        """
        if event_type not in self.event_listeners:
            logger.error(f"Unknown event type: {event_type}")
            return False
            
        self.event_listeners[event_type].append(callback)
        return True
        
    def remove_event_listener(self, event_type: str, callback: Callable) -> bool:
        """
        이벤트 리스너 제거
        
        Args:
            event_type: 이벤트 유형
            callback: 콜백 함수
            
        Returns:
            성공 여부
        """
        if event_type not in self.event_listeners:
            logger.error(f"Unknown event type: {event_type}")
            return False
            
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
            return True
        return False
        
    def _notify_listeners(self, event_type: str, data: Dict) -> None:
        """
        이벤트 리스너에 알림
        
        Args:
            event_type: 이벤트 유형
            data: 이벤트 데이터
        """
        if event_type not in self.event_listeners:
            logger.error(f"Unknown event type: {event_type}")
            return
            
        for callback in self.event_listeners[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event listener callback: {e}")
                
    def start(self) -> bool:
        """
        시스템 시작
        
        Returns:
            성공 여부
        """
        if self.running:
            logger.warning("System is already running")
            return False
            
        if not self.ecg_sensor:
            self.error_message = "No ECG sensor registered"
            logger.error(self.error_message)
            self._set_state(SystemState.ERROR)
            return False
            
        # 자극기 확인
        if len(self.stimulator.get_stimulators()) < 2:
            logger.warning("Less than two stimulators registered. Stereo stimulation won't be available.")
            
        try:
            # 센서 연결
            self._set_state(SystemState.CONNECTING)
            self.running = True
            
            # 센서 시작
            if not self.ecg_sensor.connect():
                self.error_message = "Failed to connect to ECG sensor"
                logger.error(self.error_message)
                self._set_state(SystemState.ERROR)
                self.running = False
                return False
                
            # 데이터 처리 스레드 시작
            self.data_processing_thread = threading.Thread(
                target=self._data_processing_worker,
                daemon=True
            )
            self.data_processing_thread.start()
            
            # 분석 스레드 시작
            self.analysis_thread = threading.Thread(
                target=self._analysis_worker,
                daemon=True
            )
            self.analysis_thread.start()
            
            self._set_state(SystemState.MONITORING)
            logger.info("System started successfully")
            return True
            
        except Exception as e:
            self.error_message = f"Error starting system: {e}"
            logger.error(self.error_message)
            self._set_state(SystemState.ERROR)
            self.running = False
            return False
            
    def stop(self) -> bool:
        """
        시스템 중지
        
        Returns:
            성공 여부
        """
        if not self.running:
            logger.warning("System is not running")
            return False
            
        logger.info("Stopping system...")
        self.running = False
        
        # 현재 자극 중이면 중지
        if self.state == SystemState.STIMULATING:
            self.stop_stimulation()
            
        # 센서 연결 해제
        if self.ecg_sensor:
            self.ecg_sensor.disconnect()
            
        # 스레드 종료 대기
        if self.data_processing_thread:
            self.data_processing_thread.join(timeout=2.0)
            
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
            
        self._set_state(SystemState.STOPPED)
        logger.info("System stopped")
        return True
        
    def _set_state(self, new_state: SystemState) -> None:
        """
        시스템 상태 변경
        
        Args:
            new_state: 새 상태
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"System state changed: {old_state.value} -> {new_state.value}")
            
            # 이벤트 발생
            self._notify_listeners("state_change", {
                "old_state": old_state.value,
                "new_state": new_state.value,
                "timestamp": time.time()
            })
            
    def _data_processing_worker(self) -> None:
        """ECG 데이터 처리 워커 스레드"""
        logger.info("Data processing worker started")
        
        # 센서로부터 데이터 수신 시작
        self.ecg_sensor.start_streaming(callback=self._on_ecg_data)
        
        try:
            while self.running:
                try:
                    # 큐에서 데이터 가져오기
                    ecg_data = self.ecg_data_queue.get(timeout=1.0)
                    
                    # 데이터 전처리
                    processed_data = self.ecg_processor.process(ecg_data)
                    
                    # 버퍼에 추가
                    self.ecg_buffer.extend(processed_data)
                    
                    # 버퍼 크기 제한
                    max_buffer_size = self.config.get("ecg_buffer_size", 300) * 250  # 샘플링 레이트 250Hz 가정
                    if len(self.ecg_buffer) > max_buffer_size:
                        self.ecg_buffer = self.ecg_buffer[-max_buffer_size:]
                        
                    # 큐 작업 완료 표시
                    self.ecg_data_queue.task_done()
                    
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in data processing worker: {e}")
            if self.running:
                self.error_message = f"Data processing error: {e}"
                self._set_state(SystemState.ERROR)
                
        finally:
            # 센서 스트리밍 중지
            if self.ecg_sensor:
                self.ecg_sensor.stop_streaming()
                
            logger.info("Data processing worker stopped")
            
    def _on_ecg_data(self, data: Dict) -> None:
        """
        ECG 데이터 수신 콜백
        
        Args:
            data: ECG 데이터 딕셔너리
        """
        try:
            # 큐에 데이터 추가
            if not self.ecg_data_queue.full():
                self.ecg_data_queue.put(data)
                
            # 이벤트 발생
            self._notify_listeners("ecg_data", data)
            
        except Exception as e:
            logger.error(f"Error in ECG data callback: {e}")
            
    def _analysis_worker(self) -> None:
        """HRV 분석 및 불안 예측 워커 스레드"""
        logger.info("Analysis worker started")
        
        try:
            while self.running:
                # 충분한 데이터가 있고 분석 간격이 지났는지 확인
                current_time = time.time()
                time_since_last = current_time - self.last_analysis_time
                
                if (len(self.ecg_buffer) >= 250 * 60 and  # 최소 1분 데이터
                    time_since_last >= self.analysis_interval):
                    
                    # 상태 업데이트
                    self._set_state(SystemState.ANALYZING)
                    
                    # HRV 분석
                    hrv_result = self._perform_hrv_analysis()
                    
                    # 불안 예측
                    if hrv_result:
                        anxiety_result = self._predict_anxiety(hrv_result)
                        
                        # 자동 자극 결정
                        if self.auto_stimulation and self._should_start_stimulation(anxiety_result):
                            self._auto_start_stimulation(anxiety_result)
                            
                    # 모니터링 상태로 복귀 (자극 중이 아니면)
                    if self.state != SystemState.STIMULATING:
                        self._set_state(SystemState.MONITORING)
                        
                    # 분석 시간 업데이트
                    self.last_analysis_time = current_time
                    
                # 10초마다 체크
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"Error in analysis worker: {e}")
            if self.running:
                self.error_message = f"Analysis error: {e}"
                self._set_state(SystemState.ERROR)
                
        finally:
            logger.info("Analysis worker stopped")
            
    def _perform_hrv_analysis(self) -> Dict:
        """
        HRV 분석 수행
        
        Returns:
            HRV 분석 결과 딕셔너리
        """
        try:
            # ECG 데이터에서 HRV 특성 추출
            hrv_result = self.hrv_analyzer.analyze(np.array(self.ecg_buffer))
            
            if hrv_result:
                # 결과 버퍼에 추가
                self.hrv_buffer.append(hrv_result)
                
                # 버퍼 크기 제한 (최대 100개)
                if len(self.hrv_buffer) > 100:
                    self.hrv_buffer.pop(0)
                    
                # 이벤트 발생
                self._notify_listeners("hrv_result", hrv_result)
                
                logger.info(f"HRV analysis complete: SDNN={hrv_result.get('SDNN', 0):.2f}, RMSSD={hrv_result.get('RMSSD', 0):.2f}")
                return hrv_result
                
        except Exception as e:
            logger.error(f"Error performing HRV analysis: {e}")
            
        return None
        
    def _predict_anxiety(self, hrv_features: Dict) -> Dict:
        """
        불안 수준 예측
        
        Args:
            hrv_features: HRV 특성 딕셔너리
            
        Returns:
            불안 예측 결과 딕셔너리
        """
        try:
            # 불안 예측 수행
            anxiety_result = self.anxiety_predictor.predict_anxiety(hrv_features)
            
            # 이벤트 발생
            self._notify_listeners("anxiety_prediction", anxiety_result)
            
            anxiety_level = anxiety_result.get("anxiety_level", AnxietyLevel.NORMAL)
            anxiety_score = anxiety_result.get("anxiety_score", 0.0)
            
            logger.info(f"Anxiety prediction: level={anxiety_level.name}, score={anxiety_score:.1f}")
            return anxiety_result
            
        except Exception as e:
            logger.error(f"Error predicting anxiety: {e}")
            return None
            
    def _should_start_stimulation(self, anxiety_result: Dict) -> bool:
        """
        자극 시작 여부 결정
        
        Args:
            anxiety_result: 불안 예측 결과
            
        Returns:
            자극 시작 여부
        """
        # 이미 자극 중이면 시작하지 않음
        if self.state == SystemState.STIMULATING:
            return False
            
        # 불안 점수가 임계값을 넘으면 자극 시작
        anxiety_score = anxiety_result.get("anxiety_score", 0.0)
        anxiety_level = anxiety_result.get("anxiety_level", AnxietyLevel.NORMAL)
        
        # 중등도 이상의 불안 또는 점수가 임계값 초과
        if (anxiety_level.value >= AnxietyLevel.MODERATE.value or
            anxiety_score >= self.anxiety_threshold):
            
            # 자극 권장 여부 확인
            recommendation = self.anxiety_predictor.get_intervention_recommendation()
            
            return recommendation.get("stimulation_recommended", False)
            
        return False
        
    def _auto_start_stimulation(self, anxiety_result: Dict) -> None:
        """
        자동 자극 시작
        
        Args:
            anxiety_result: 불안 예측 결과
        """
        # 자극 파라미터 결정
        recommendation = self.anxiety_predictor.get_intervention_recommendation()
        stim_params = recommendation.get("stimulation_params", {})
        
        # 불안 수준에 따른 자극 파라미터 설정
        anxiety_level = anxiety_result.get("anxiety_level", AnxietyLevel.NORMAL)
        
        # 자극 시작
        self.start_stimulation(stim_params)
        
    def start_stimulation(self, params: Optional[Dict] = None) -> bool:
        """
        자극 시작
        
        Args:
            params: 자극 파라미터 딕셔너리 (None인 경우 기본값 사용)
            
        Returns:
            성공 여부
        """
        if not self.running:
            logger.warning("Cannot start stimulation when system is not running")
            return False
            
        if self.state == SystemState.STIMULATING:
            logger.warning("Stimulation is already active")
            return False
            
        # 기본 파라미터 설정
        stim_params = self.config.get("stimulation_default_params", {}).copy()
        
        # 사용자 파라미터로 업데이트
        if params:
            stim_params.update(params)
            
        # 파형 타입 변환
        if "waveform" in stim_params and isinstance(stim_params["waveform"], str):
            stim_params["waveform"] = WaveformType(stim_params["waveform"])
            
        # 자극 시작
        if self.stimulator.start_stimulation(stim_params):
            self._set_state(SystemState.STIMULATING)
            
            # 이벤트 발생
            self._notify_listeners("stimulation_start", {
                "params": stim_params,
                "timestamp": time.time()
            })
            
            logger.info(f"Stimulation started with params: {stim_params}")
            return True
        else:
            logger.error("Failed to start stimulation")
            return False
            
    def stop_stimulation(self) -> bool:
        """
        자극 중지
        
        Returns:
            성공 여부
        """
        if self.state != SystemState.STIMULATING:
            logger.warning("Stimulation is not active")
            return False
            
        # 자극 중지
        if self.stimulator.stop_stimulation():
            # 모니터링 상태로 복귀
            self._set_state(SystemState.MONITORING)
            
            # 이벤트 발생
            self._notify_listeners("stimulation_stop", {
                "timestamp": time.time()
            })
            
            logger.info("Stimulation stopped")
            return True
        else:
            logger.error("Failed to stop stimulation")
            return False
            
    def set_stimulation_phase_delay(self, delay: float) -> bool:
        """
        자극 위상 지연 설정
        
        Args:
            delay: 위상 지연 (초)
            
        Returns:
            성공 여부
        """
        result = self.stimulator.set_phase_delay(delay)
        if result:
            logger.info(f"Stimulation phase delay set to {delay} seconds")
        return result
        
    def set_stimulation_balance(self, balance_params: Dict[str, float]) -> bool:
        """
        자극 밸런스 설정
        
        Args:
            balance_params: 자극기별 강도 밸런스 딕셔너리
            
        Returns:
            성공 여부
        """
        result = self.stimulator.set_balance(balance_params)
        if result:
            logger.info(f"Stimulation balance set to {balance_params}")
        return result
        
    def set_auto_stimulation(self, enabled: bool) -> None:
        """
        자동 자극 활성화 설정
        
        Args:
            enabled: 활성화 여부
        """
        self.auto_stimulation = enabled
        logger.info(f"Auto stimulation {'enabled' if enabled else 'disabled'}")
        
    def set_anxiety_threshold(self, threshold: float) -> None:
        """
        불안 임계값 설정
        
        Args:
            threshold: 불안 임계값 (0-100)
        """
        self.anxiety_threshold = max(0.0, min(100.0, threshold))
        logger.info(f"Anxiety threshold set to {self.anxiety_threshold}")
        
    def get_system_status(self) -> Dict:
        """
        시스템 상태 정보 반환
        
        Returns:
            상태 정보 딕셔너리
        """
        anxiety_level = AnxietyLevel.NORMAL
        anxiety_score = 0.0
        
        # 마지막 불안 예측 결과
        if self.anxiety_predictor.last_prediction:
            anxiety_level = self.anxiety_predictor.last_prediction.get("anxiety_level", AnxietyLevel.NORMAL)
            anxiety_score = self.anxiety_predictor.last_prediction.get("anxiety_score", 0.0)
            
        # 자극기 상태
        stimulators = {}
        for stim_id in self.stimulator.get_stimulators():
            stimulators[stim_id] = self.stimulator.get_stimulator_status(stim_id)
            
        return {
            "state": self.state.value,
            "running": self.running,
            "error_message": self.error_message,
            "ecg_sensor_connected": self.ecg_sensor.is_connected() if self.ecg_sensor else False,
            "ecg_buffer_size": len(self.ecg_buffer),
            "last_analysis_time": self.last_analysis_time,
            "anxiety_level": anxiety_level.name,
            "anxiety_score": anxiety_score,
            "auto_stimulation": self.auto_stimulation,
            "anxiety_threshold": self.anxiety_threshold,
            "stimulators": stimulators,
            "timestamp": time.time()
        }
        
    def save_data(self, path: Optional[str] = None) -> bool:
        """
        현재 데이터 저장
        
        Args:
            path: 저장 경로 (None인 경우 설정의 기본 경로 사용)
            
        Returns:
            성공 여부
        """
        if not path:
            path = self.config.get("data_save_path", "./data/")
            
        try:
            import os
            os.makedirs(path, exist_ok=True)
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # ECG 데이터 저장
            if self.ecg_buffer:
                ecg_path = os.path.join(path, f"ecg_{timestamp}.npy")
                np.save(ecg_path, np.array(self.ecg_buffer))
                
            # HRV 데이터 저장
            if self.hrv_buffer:
                hrv_path = os.path.join(path, f"hrv_{timestamp}.json")
                with open(hrv_path, 'w') as f:
                    json.dump(self.hrv_buffer, f)
                    
            # 불안 예측 이력 저장
            if self.anxiety_predictor.prediction_history:
                anxiety_path = os.path.join(path, f"anxiety_{timestamp}.json")
                with open(anxiety_path, 'w') as f:
                    json.dump(self.anxiety_predictor.prediction_history, f)
                    
            logger.info(f"Data saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False


# 시스템 인스턴스 생성 및 실행 예
if __name__ == "__main__":
    # 로깅 레벨 설정
    logging.getLogger().setLevel(logging.INFO)
    
    # 컨트롤러 생성
    controller = AnxietyPreventionController()
    
    # 이벤트 리스너 예시
    def on_anxiety_prediction(data):
        level = data.get("anxiety_level")
        score = data.get("anxiety_score", 0.0)
        print(f"불안 수준: {level.name}, 점수: {score:.1f}")
        
    controller.add_event_listener("anxiety_prediction", on_anxiety_prediction)
    
    # 참고: 실제 구현에서는 센서와 자극기를 등록해야 함
    # from ecg_sensor.bluetooth_sensor import BluetoothECGSensor
    # sensor = BluetoothECGSensor()
    # controller.register_ecg_sensor(sensor)
    # 
    # from stimulation_controller.bluetooth_stimulator import BluetoothStimulator
    # stim1 = BluetoothStimulator("device1_mac")
    # stim2 = BluetoothStimulator("device2_mac")
    # controller.register_stimulator("stimulator1", stim1)
    # controller.register_stimulator("stimulator2", stim2)
    
    # 시스템 시작
    # controller.start()
    
    print("불안장애 예방 시스템 컨트롤러 모듈")
    print("실제 실행을 위해 메인 애플리케이션에서 필요한 센서와 자극기를 등록해야 합니다.")
