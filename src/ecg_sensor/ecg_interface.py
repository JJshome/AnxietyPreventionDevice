import time
import threading
import logging
import numpy as np
import random
from enum import Enum

logger = logging.getLogger(__name__)

class SensorStatus(Enum):
    """센서 상태 열거형"""
    DISCONNECTED = "disconnected"  # 연결 해제
    CONNECTING = "connecting"     # 연결 중
    CONNECTED = "connected"       # 연결됨
    ERROR = "error"               # 오류

class ECGInterface:
    """
    심전도(ECG) 센서와의 인터페이스를 제공하는 클래스.
    
    웨어러블 ECG 센서와의 통신을 관리하고, 데이터를 수집합니다.
    시뮤레이션 모드에서는 가상 ECG 데이터를 생성합니다.
    """
    
    def __init__(self, sampling_rate=256, buffer_size=76800, bluetooth_enabled=True):
        """
        ECGInterface 초기화
        
        매개변수:
            sampling_rate (int): 샘플링 레이트 (Hz)
            buffer_size (int): 버퍼 크기
            bluetooth_enabled (bool): 블루투스 활성화 여부
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.bluetooth_enabled = bluetooth_enabled
        
        self.status = SensorStatus.DISCONNECTED
        self.device = None
        self.running = False
        self.data_thread = None
        
        # 데이터 버퍼
        self.data_buffer = []
        self.last_read_index = 0
        
        # 시뮤레이션 모드 (실제 센서가 연결되지 않은 경우)
        self.simulation_mode = not bluetooth_enabled
        
        logger.info(f"ECG 인터페이스가 초기화되었습니다. 시뮤레이션 모드: {self.simulation_mode}")
    
    def connect(self):
        """
        ECG 센서에 연결합니다.
        
        반환값:
            bool: 연결 성공 여부
        """
        if self.status == SensorStatus.CONNECTED:
            logger.warning("이미 센서에 연결되어 있습니다.")
            return True
        
        try:
            self.status = SensorStatus.CONNECTING
            
            if self.simulation_mode:
                logger.info("시뮤레이션 모드에서 가상 ECG 센서에 연결합니다.")
                # 시뮤레이션 모드에서는 가상 연결만 수행
                time.sleep(1)  # 연결 지연 시뮤레이션
                self.device = "Simulated ECG Sensor"
            else:
                # 블루투스 ECG 센서 연결 (실제 구현 필요)
                logger.info("블루투스 ECG 센서 연결 시도 중...")
                # 센서 스캔, 연결 설정 등
                # 여기서는 예시만 제공하며, 실제 하드웨어와 통신하려면 구현 필요
                time.sleep(2)  # 연결 지연 시뮤레이션
                self.device = "ECG Sensor BLE"
            
            self.status = SensorStatus.CONNECTED
            self.running = True
            
            # 데이터 수집 스레드 시작
            self.data_thread = threading.Thread(
                target=self._data_collection_worker
            )
            self.data_thread.daemon = True
            self.data_thread.start()
            
            logger.info(f"ECG 센서 '{self.device}'에 연결되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"ECG 센서 연결 중 오류 발생: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def disconnect(self):
        """
        ECG 센서와의 연결을 해제합니다.
        
        반환값:
            bool: 연결 해제 성공 여부
        """
        if self.status != SensorStatus.CONNECTED:
            logger.warning("연결된 센서가 없습니다.")
            return True
        
        try:
            # 스레드 중지
            self.running = False
            
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=1.0)
            
            if not self.simulation_mode:
                # 블루투스 연결 해제 (실제 구현 필요)
                pass
            
            self.status = SensorStatus.DISCONNECTED
            self.device = None
            logger.info("ECG 센서와의 연결이 해제되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"ECG 센서 연결 해제 중 오류 발생: {e}")
            self.status = SensorStatus.ERROR
            return False
    
    def read_data(self):
        """
        수집된 ECG 데이터를 읽습니다.
        
        반환값:
            ndarray: 수집된 ECG 데이터 또는 None
        """
        if self.status != SensorStatus.CONNECTED:
            return None
        
        if not self.data_buffer:
            return None
        
        # 버퍼에서 새로운 데이터만 반환
        if self.last_read_index >= len(self.data_buffer):
            return None
        
        data = self.data_buffer[self.last_read_index:]
        self.last_read_index = len(self.data_buffer)
        
        return np.array(data)
    
    def get_status(self):
        """
        현재 센서 상태를 반환합니다.
        
        반환값:
            dict: 센서 상태 정보
        """
        return {
            "status": self.status.value,
            "device": self.device,
            "simulation_mode": self.simulation_mode,
            "buffer_size": len(self.data_buffer),
            "sampling_rate": self.sampling_rate
        }
    
    def _data_collection_worker(self):
        """
        ECG 데이터 수집 워커 함수
        """
        logger.info("ECG 데이터 수집이 시작되었습니다.")
        
        # 시간 간격 계산 (샘플링 레이트에 따라)
        interval = 1.0 / self.sampling_rate
        last_time = time.time()
        
        # 시뮤레이션을 위한 변수
        simulation_phase = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # 샘플링 레이트에 따라 데이터 수집
                if current_time - last_time >= interval:
                    if self.simulation_mode:
                        # 가상 ECG 데이터 생성
                        new_data = self._generate_simulated_ecg(simulation_phase)
                        simulation_phase += interval
                    else:
                        # 실제 센서에서 데이터 읽기 (실제 구현 필요)
                        new_data = 0  # 임시 값
                    
                    # 버퍼에 데이터 추가
                    self.data_buffer.append(new_data)
                    
                    # 버퍼 크기 제한
                    if len(self.data_buffer) > self.buffer_size:
                        self.data_buffer = self.data_buffer[-self.buffer_size:]
                    
                    last_time = current_time
                
                # 처리 시간 최적화를 위한 잠시 대기
                time.sleep(interval / 10)
        
        except Exception as e:
            logger.error(f"ECG 데이터 수집 중 오류 발생: {e}")
            self.status = SensorStatus.ERROR
    
    def _generate_simulated_ecg(self, phase):
        """
        시뮤레이션된 ECG 데이터를 생성합니다.
        
        매개변수:
            phase (float): 시뮤레이션 위상
            
        반환값:
            float: 생성된 ECG 값
        """
        # 기본 심박수 (60-80 BPM)
        heart_rate = 70 + 10 * np.sin(phase / 60)  # 심박수가 천천히 변화
        beat_period = 60.0 / heart_rate
        
        # 현재 주기 내에서의 위치
        t = phase % beat_period / beat_period
        
        # P파, QRS 복합체, T파를 모방한 ECG 생성
        p_wave = 0.25 * np.exp(-((t - 0.2) ** 2) / 0.01)
        qrs_complex = 1.0 * np.exp(-((t - 0.4) ** 2) / 0.002) - 0.3 * np.exp(-((t - 0.38) ** 2) / 0.003) - 0.2 * np.exp(-((t - 0.43) ** 2) / 0.003)
        t_wave = 0.35 * np.exp(-((t - 0.7) ** 2) / 0.02)
        
        # 기준선 변동 및 노이즈 추가
        baseline = 0.05 * np.sin(phase / 2) + 0.03 * np.sin(phase / 5)
        noise = 0.03 * random.random() - 0.015
        
        # 심전도 신호 조합
        ecg = p_wave + qrs_complex + t_wave + baseline + noise
        
        return ecg
