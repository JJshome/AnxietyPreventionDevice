import time
import threading
import logging
import numpy as np
from enum import Enum, auto
from collections import deque

from src.hrv_analyzer import HRVAnalyzer
from src.anxiety_predictor import AnxietyPredictor
from src.stim_controller import StimController, WaveformType, PhaseType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnxietyPreventionController')

class SystemState(Enum):
    """시스템 상태"""
    IDLE = auto()           # 대기 상태
    MONITORING = auto()     # HRV 모니터링 중
    PREDICTING = auto()     # 불안 예측 중
    STIMULATING = auto()    # 자극 전달 중
    CALIBRATING = auto()    # 시스템 교정 중
    ERROR = auto()          # 오류 상태

class OperationMode(Enum):
    """작동 모드"""
    AUTOMATIC = auto()      # 자동 모드 (예측 기반)
    MANUAL = auto()         # 수동 모드 (사용자 제어)
    PREVENTION = auto()     # 예방 모드 (낮은 임계값)
    TRAINING = auto()       # 훈련 모드 (고정 스케줄)

class AnxietyPreventionController:
    """
    불안장애 예방 시스템 제어 모듈
    
    특허 10-2022-0007209에 따른 불안장애 예방장치 제어 시스템.
    ECG 신호 분석, 불안 예측, 그리고 저주파 자극기 제어를 통합합니다.
    """
    
    def __init__(self, 
                 sampling_rate=256, 
                 anxiety_threshold=0.6,
                 stimulation_cooldown=1800,  # 30분(초)
                 window_size=60,            # 60초
                 auto_start=False):
        """
        AnxietyPreventionController 초기화
        
        Args:
            sampling_rate (int): ECG 신호 샘플링 레이트 (Hz)
            anxiety_threshold (float): 불안 상태 판단 임계값 (0-1)
            stimulation_cooldown (int): 자극 이후 대기 시간 (초)
            window_size (int): HRV 분석 윈도우 크기 (초)
            auto_start (bool): 초기화 후 자동 시작 여부
        """
        self.sampling_rate = sampling_rate
        self.anxiety_threshold = anxiety_threshold
        self.stimulation_cooldown = stimulation_cooldown
        self.window_size = window_size
        
        # 시스템 상태
        self.state = SystemState.IDLE
        self.operation_mode = OperationMode.AUTOMATIC
        self.running = False
        self.last_stimulation_time = 0
        
        # 부부 모듈 초기화
        self.hrv_analyzer = HRVAnalyzer(sampling_rate=sampling_rate, window_size=window_size)
        self.anxiety_predictor = AnxietyPredictor(threshold=anxiety_threshold)
        self.stim_controller = StimController(num_stimulators=2)
        
        # 데이터 저장소
        self.ecg_buffer = deque(maxlen=sampling_rate * window_size * 2)  # 최대 2분 데이터
        self.hrv_features = deque(maxlen=100)  # 최근 100개 HRV 특성
        self.anxiety_scores = deque(maxlen=100)  # 최근 100개 불안 점수
        self.stimulation_history = deque(maxlen=50)  # 최근 50개 자극 기록
        
        # 스레드
        self.worker_thread = None
        self.lock = threading.Lock()
        
        # 자동 시작
        if auto_start:
            self.start()
        
        logger.info("AnxietyPreventionController initialized")
    
    def start(self):
        """
        시스템 시작
        
        Returns:
            bool: 성공 여부
        """
        if self.running:
            logger.warning("System already running")
            return False
        
        logger.info("Starting AnxietyPreventionController")
        self.running = True
        self.state = SystemState.MONITORING
        
        # 작업 스레드 시작
        self.worker_thread = threading.Thread(target=self._main_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        return True
    
    def stop(self):
        """
        시스템 중지
        
        Returns:
            bool: 성공 여부
        """
        if not self.running:
            logger.warning("System already stopped")
            return False
        
        logger.info("Stopping AnxietyPreventionController")
        self.running = False
        
        # 활성화된 자극기 중지
        if self.state == SystemState.STIMULATING:
            self.stim_controller.stop_stimulation()
        
        # 작업 스레드 종료 대기
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        self.state = SystemState.IDLE
        return True
    
    def _main_loop(self):
        """
        주요 작업 루프
        """
        last_analysis_time = 0
        last_prediction_time = 0
        analysis_interval = 10  # HRV 분석 간격 (초)
        prediction_interval = 30  # 불안 예측 간격 (초)
        
        while self.running:
            try:
                current_time = time.time()
                
                # HRV 분석
                if current_time - last_analysis_time >= analysis_interval and len(self.ecg_buffer) > self.sampling_rate * 10:
                    self.state = SystemState.MONITORING
                    with self.lock:
                        ecg_data = np.array(list(self.ecg_buffer))
                    
                    # HRV 분석 수행
                    hrv_features = self.hrv_analyzer.calculate_hrv_features(ecg_data)
                    if hrv_features:  # 분석 결과가 유효한 경우
                        hrv_features['timestamp'] = current_time
                        self.hrv_features.append(hrv_features)
                        logger.debug(f"HRV analysis completed: SDNN={hrv_features.get('sdnn', 0):.2f}, "
                                    f"RMSSD={hrv_features.get('rmssd', 0):.2f}")
                    
                    last_analysis_time = current_time
                
                # 불안 예측
                if current_time - last_prediction_time >= prediction_interval and len(self.hrv_features) > 0:
                    self.state = SystemState.PREDICTING
                    
                    # 최근 HRV 특성으로 불안 점수 예측
                    latest_features = self.hrv_features[-1]
                    anxiety_score = self.anxiety_predictor.predict(latest_features, return_probability=True)
                    
                    # 점수 저장
                    self.anxiety_scores.append({
                        'timestamp': current_time,
                        'score': anxiety_score,
                        'level': self.anxiety_predictor.get_anxiety_level(anxiety_score)
                    })
                    
                    logger.info(f"Anxiety prediction: score={anxiety_score:.4f}, "
                               f"threshold={self.anxiety_threshold:.4f}")
                    
                    # 자동 모드에서 임계값 초과시 자극 시갑
                    if self.operation_mode == OperationMode.AUTOMATIC and anxiety_score >= self.anxiety_threshold:
                        time_since_last_stim = current_time - self.last_stimulation_time
                        if time_since_last_stim >= self.stimulation_cooldown:
                            self._start_auto_stimulation(anxiety_score)
                    elif self.operation_mode == OperationMode.PREVENTION and anxiety_score >= self.anxiety_threshold * 0.7:
                        time_since_last_stim = current_time - self.last_stimulation_time
                        if time_since_last_stim >= self.stimulation_cooldown:
                            self._start_auto_stimulation(anxiety_score, prevention_mode=True)
                    
                    last_prediction_time = current_time
                
                # 훈련 모드 처리
                if self.operation_mode == OperationMode.TRAINING:
                    self._handle_training_mode()
                
                # 잠시 대기
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.state = SystemState.ERROR
                time.sleep(1)  # 오류 발생 시 짧게 대기
        
        logger.info("Main loop stopped")
    
    def _start_auto_stimulation(self, anxiety_score, prevention_mode=False):
        """
        자동 자극 시작
        
        Args:
            anxiety_score (float): 예측된 불안 점수
            prevention_mode (bool): 예방 모드 여부
        """
        # 자극 파라미터 설정
        stim_params = {}
        
        if prevention_mode:
            # 예방 모드: 낮은 강도, 오랜 시간
            duration = 180  # 3분
            stim_params = {
                'frequency': 5.0,  # 낮은 주파수
                'amplitude': 0.6,
                'waveform_type': WaveformType.SINE.name,
                'phase_type': PhaseType.BIPHASIC.name,
                'phase_delay': 0.5
            }
        else:
            # 일반 모드: 불안 정도에 따라 주파수 조절
            duration = 120  # 2분
            
            # 불안 점수에 따라 주파수 조절 (0.6-1.0 -> 10-25Hz)
            freq_factor = (anxiety_score - self.anxiety_threshold) / (1.0 - self.anxiety_threshold)
            freq_factor = min(1.0, max(0.0, freq_factor))
            frequency = 10.0 + freq_factor * 15.0
            
            stim_params = {
                'frequency': frequency,
                'amplitude': 0.8,
                'waveform_type': WaveformType.SINE.name,
                'phase_type': PhaseType.BIPHASIC.name,
                'phase_delay': 0.3 + freq_factor * 0.4  # 0.3-0.7초
            }
        
        # 자극 시작
        stim_result = self.stim_controller.start_stimulation(duration=duration, custom_params=stim_params)
        
        if stim_result:
            self.state = SystemState.STIMULATING
            self.last_stimulation_time = time.time()
            
            # 자극 기록 저장
            stim_record = {
                'timestamp': self.last_stimulation_time,
                'duration': duration,
                'anxiety_score': anxiety_score,
                'mode': 'prevention' if prevention_mode else 'treatment',
                'parameters': stim_params
            }
            self.stimulation_history.append(stim_record)
            
            logger.info(f"Auto stimulation started: anxiety_score={anxiety_score:.4f}, "
                       f"frequency={stim_params['frequency']:.1f}Hz, duration={duration}s")
        else:
            logger.warning("Failed to start auto stimulation")
    
    def _handle_training_mode(self):
        """
        훈련 모드 처리
        """
        # 현재 은 훈련 모드 구현은 싶편이지만 추후 구현 예정
        pass
    
    def process_ecg_data(self, ecg_data):
        """
        실시간 ECG 데이터 처리
        
        Args:
            ecg_data (np.ndarray): ECG 신호 데이터
            
        Returns:
            bool: 성공 여부
        """
        if not self.running:
            logger.warning("Cannot process ECG data: system not running")
            return False
        
        # 신호 처리
        with self.lock:
            # 버퍼에 데이터 추가
            self.ecg_buffer.extend(ecg_data)
        
        return True
    
    def manual_stimulation(self, duration=60, parameters=None):
        """
        수동 자극 시작
        
        Args:
            duration (int): 자극 지속 시간 (초)
            parameters (dict): 자극 매개변수
            
        Returns:
            bool: 성공 여부
        """
        if not self.running:
            logger.warning("Cannot start manual stimulation: system not running")
            return False
        
        # 자극 시작
        stim_result = self.stim_controller.start_stimulation(duration=duration, custom_params=parameters)
        
        if stim_result:
            self.state = SystemState.STIMULATING
            self.last_stimulation_time = time.time()
            
            # 자극 기록 저장
            stim_record = {
                'timestamp': self.last_stimulation_time,
                'duration': duration,
                'anxiety_score': self.anxiety_scores[-1]['score'] if self.anxiety_scores else 0.0,
                'mode': 'manual',
                'parameters': parameters or self.stim_controller.get_settings()
            }
            self.stimulation_history.append(stim_record)
            
            logger.info(f"Manual stimulation started: duration={duration}s")
            return True
        else:
            logger.warning("Failed to start manual stimulation")
            return False
    
    def stop_stimulation(self):
        """
        자극 중지
        
        Returns:
            bool: 성공 여부
        """
        if self.state != SystemState.STIMULATING:
            logger.warning("No active stimulation to stop")
            return False
        
        # 자극 중지
        if self.stim_controller.stop_stimulation():
            self.state = SystemState.MONITORING
            logger.info("Stimulation stopped")
            return True
        else:
            logger.warning("Failed to stop stimulation")
            return False
    
    def set_anxiety_threshold(self, threshold):
        """
        불안 임계값 설정
        
        Args:
            threshold (float): 새 임계값 (0-1)
            
        Returns:
            bool: 성공 여부
        """
        if 0 <= threshold <= 1:
            self.anxiety_threshold = threshold
            self.anxiety_predictor.set_threshold(threshold)
            logger.info(f"Anxiety threshold set to {threshold}")
            return True
        else:
            logger.warning(f"Invalid anxiety threshold: {threshold}, must be between 0 and 1")
            return False
    
    def set_operation_mode(self, mode):
        """
        작동 모드 설정
        
        Args:
            mode (OperationMode or str): 새 작동 모드
            
        Returns:
            bool: 성공 여부
        """
        if isinstance(mode, str):
            mode = mode.upper()
            if hasattr(OperationMode, mode):
                mode = getattr(OperationMode, mode)
            else:
                logger.warning(f"Unknown operation mode: {mode}")
                return False
        
        self.operation_mode = mode
        logger.info(f"Operation mode set to {mode.name}")
        return True
    
    def set_stimulation_parameters(self, parameters):
        """
        자극 매개변수 설정
        
        Args:
            parameters (dict): 자극 매개변수
            
        Returns:
            bool: 성공 여부
        """
        return self.stim_controller.set_stimulation_parameters(parameters)
    
    def get_status(self):
        """
        시스템 상태 정보 조회
        
        Returns:
            dict: 시스템 상태 정보
        """
        status = {
            'state': self.state.name,
            'operation_mode': self.operation_mode.name,
            'running': self.running,
            'anxiety_threshold': self.anxiety_threshold,
            'stimulation_cooldown': self.stimulation_cooldown,
            'last_stimulation_time': self.last_stimulation_time,
            'time_since_last_stimulation': time.time() - self.last_stimulation_time,
            'stimulation_available': (time.time() - self.last_stimulation_time) >= self.stimulation_cooldown,
        }
        
        # 최근 불안 점수
        if self.anxiety_scores:
            status['anxiety_score'] = self.anxiety_scores[-1]['score']
            status['anxiety_level'] = self.anxiety_scores[-1]['level']
        else:
            status['anxiety_score'] = 0.0
            status['anxiety_level'] = 'unknown'
        
        # 현재 자극 정보
        if self.state == SystemState.STIMULATING:
            active_stim = self.stim_controller.get_settings()
            status['stimulation'] = {
                'active': True,
                'start_time': self.last_stimulation_time,
                'elapsed_time': time.time() - self.last_stimulation_time,
                'frequency': active_stim.get('frequency', 0),
                'amplitude': active_stim.get('amplitude', 0),
                'waveform_type': active_stim.get('waveform_type', 'UNKNOWN'),
                'phase_delay': active_stim.get('phase_delay', 0),
            }
            
            # 자극 기록에서 현재 자극 정보 가져오기
            if self.stimulation_history:
                latest_stim = self.stimulation_history[-1]
                status['stimulation']['duration'] = latest_stim.get('duration', 60)
                status['stimulation']['mode'] = latest_stim.get('mode', 'unknown')
        else:
            status['stimulation'] = {'active': False}
        
        return status
    
    def get_hrv_data(self, limit=None):
        """
        HRV 데이터 조회
        
        Args:
            limit (int, optional): 최대 개수
            
        Returns:
            list: HRV 데이터 목록
        """
        hrv_data = list(self.hrv_features)
        if limit is not None and limit > 0:
            return hrv_data[-limit:]
        return hrv_data
    
    def get_anxiety_scores(self, limit=None):
        """
        불안 점수 조회
        
        Args:
            limit (int, optional): 최대 개수
            
        Returns:
            list: 불안 점수 목록
        """
        scores = list(self.anxiety_scores)
        if limit is not None and limit > 0:
            return scores[-limit:]
        return scores
    
    def get_stimulation_history(self, limit=None):
        """
        자극 기록 조회
        
        Args:
            limit (int, optional): 최대 개수
            
        Returns:
            list: 자극 기록 목록
        """
        history = list(self.stimulation_history)
        if limit is not None and limit > 0:
            return history[-limit:]
        return history
    
    def set_stimulation_cooldown(self, cooldown):
        """
        자극 재시작 대기시간 설정
        
        Args:
            cooldown (int): 대기시간 (초)
            
        Returns:
            bool: 성공 여부
        """
        if cooldown >= 0:
            self.stimulation_cooldown = cooldown
            logger.info(f"Stimulation cooldown set to {cooldown} seconds")
            return True
        else:
            logger.warning(f"Invalid cooldown: {cooldown}, must be non-negative")
            return False
    
    def calibrate(self):
        """
        시스템 교정
        
        Returns:
            bool: 성공 여부
        """
        if not self.running:
            logger.warning("Cannot calibrate: system not running")
            return False
        
        if self.state == SystemState.STIMULATING:
            logger.warning("Cannot calibrate during stimulation")
            return False
        
        prev_state = self.state
        self.state = SystemState.CALIBRATING
        
        try:
            logger.info("Starting system calibration...")
            # 교정 작업 수행
            # (실제 교정 로직을 구현해야 함)
            
            logger.info("Calibration completed successfully")
            self.state = SystemState.MONITORING
            return True
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.state = prev_state
            return False
    
    def reset(self):
        """
        시스템 초기화
        
        Returns:
            bool: 성공 여부
        """
        was_running = self.running
        
        # 시스템 중지
        if was_running:
            self.stop()
        
        # 데이터 초기화
        self.ecg_buffer.clear()
        self.hrv_features.clear()
        self.anxiety_scores.clear()
        self.stimulation_history.clear()
        
        # 상태 초기화
        self.state = SystemState.IDLE
        self.last_stimulation_time = 0
        
        logger.info("System reset completed")
        
        # 필요시 재시작
        if was_running:
            self.start()
        
        return True