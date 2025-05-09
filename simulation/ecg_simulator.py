"""
ECG 신호 시뮬레이터

웨어러블 심전도 센서의 ECG 신호를 시뮬레이션하는 모듈입니다.
실제 ECG 데이터 세트를 기반으로 다양한 심박 패턴을 생성합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import threading
import time
import logging
import os
import random
from enum import Enum

logger = logging.getLogger(__name__)


class ECGPatternType(Enum):
    """ECG 패턴 유형"""
    NORMAL = "normal"               # 정상 심박 패턴
    ELEVATED_HR = "elevated_hr"     # 높은 심박수
    REDUCED_HRV = "reduced_hrv"     # 감소된 심박변이도
    ANXIETY = "anxiety"             # 불안 상태 패턴
    ARRHYTHMIA = "arrhythmia"       # 부정맥 패턴


class ECGSimulator:
    """
    웨어러블 ECG 센서의 심전도 신호를 시뮬레이션하는 클래스
    """
    
    def __init__(self, sampling_rate=256, noise_level=0.1):
        """
        ECGSimulator 초기화
        
        매개변수:
            sampling_rate (int): 샘플링 레이트 (Hz)
            noise_level (float): 노이즈 레벨 (0-1)
        """
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        
        # 기본 ECG 템플릿 로드
        self.templates = self._load_templates()
        
        # 시뮬레이션 상태 변수
        self.running = False
        self.simulation_thread = None
        self.listeners = []
        
        # 현재 패턴 설정
        self.current_pattern = ECGPatternType.NORMAL
        self.current_heart_rate = 70  # bpm
        self.heart_rate_variability = 0.1  # 심박 변이 수준 (0-1)
        
        # 생성된 데이터 저장
        self.generated_data = []
        self.timestamps = []
        
        logger.info("ECG 시뮬레이터가 초기화되었습니다.")
    
    def _load_templates(self):
        """
        ECG 신호 템플릿을 로드합니다.
        실제 구현에서는 파일에서 로드하거나 더 정교한 모델을 사용할 수 있습니다.
        
        반환값:
            dict: 패턴 유형별 ECG 템플릿
        """
        # 간단한 ECG PQRST 파형 생성
        def generate_pqrst(duration=0.8, sampling_rate=256):
            """
            PQRST 파형을 생성합니다.
            
            매개변수:
                duration (float): 파형 지속 시간 (초)
                sampling_rate (int): 샘플링 레이트 (Hz)
                
            반환값:
                ndarray: PQRST 파형
            """
            t = np.arange(0, duration, 1.0/sampling_rate)
            n_samples = len(t)
            
            # 가우시안 함수를 사용한 파형 생성
            p_wave = -0.2 * np.exp(-((t - 0.1)**2) / 0.005)
            q_wave = -0.5 * np.exp(-((t - 0.2)**2) / 0.002)
            r_wave = 3.0 * np.exp(-((t - 0.25)**2) / 0.001)
            s_wave = -0.7 * np.exp(-((t - 0.3)**2) / 0.002)
            t_wave = 0.5 * np.exp(-((t - 0.5)**2) / 0.01)
            
            # 파형 합성
            pqrst = p_wave + q_wave + r_wave + s_wave + t_wave
            
            # 기준선 조정
            baseline = np.zeros_like(pqrst)
            ecg_wave = pqrst + baseline
            
            return ecg_wave
        
        # 다양한 패턴에 대한 템플릿 생성
        templates = {}
        
        # 정상 패턴
        templates[ECGPatternType.NORMAL] = generate_pqrst(
            duration=0.8, sampling_rate=self.sampling_rate
        )
        
        # 높은 심박수 패턴 (더 짧은 주기)
        templates[ECGPatternType.ELEVATED_HR] = generate_pqrst(
            duration=0.6, sampling_rate=self.sampling_rate
        )
        
        # 불안 패턴 (더 높은 R파, 더 짧은 주기)
        anxiety_t = np.arange(0, 0.6, 1.0/self.sampling_rate)
        p_wave = -0.2 * np.exp(-((anxiety_t - 0.1)**2) / 0.004)
        q_wave = -0.6 * np.exp(-((anxiety_t - 0.2)**2) / 0.001)
        r_wave = 3.5 * np.exp(-((anxiety_t - 0.25)**2) / 0.0008)  # 더 높은 R파
        s_wave = -0.9 * np.exp(-((anxiety_t - 0.3)**2) / 0.001)
        t_wave = 0.4 * np.exp(-((anxiety_t - 0.45)**2) / 0.008)
        templates[ECGPatternType.ANXIETY] = (
            p_wave + q_wave + r_wave + s_wave + t_wave
        )
        
        # 부정맥 패턴 (불규칙한 R파 높이와 간격)
        arrhythmia_t = np.arange(0, 0.9, 1.0/self.sampling_rate)
        p_wave = -0.15 * np.exp(-((arrhythmia_t - 0.12)**2) / 0.006)
        q_wave = -0.4 * np.exp(-((arrhythmia_t - 0.22)**2) / 0.003)
        r_wave = 2.7 * np.exp(-((arrhythmia_t - 0.28)**2) / 0.0015)
        s_wave = -0.6 * np.exp(-((arrhythmia_t - 0.34)**2) / 0.003)
        t_wave = 0.4 * np.exp(-((arrhythmia_t - 0.6)**2) / 0.015)
        templates[ECGPatternType.ARRHYTHMIA] = (
            p_wave + q_wave + r_wave + s_wave + t_wave
        )
        
        # 감소된 HRV 패턴 (정상 템플릿과 동일하나 생성 시 심박 간격 변이가 줄어듬)
        templates[ECGPatternType.REDUCED_HRV] = templates[ECGPatternType.NORMAL]
        
        return templates
    
    def add_listener(self, callback):
        """
        ECG 데이터 리스너를 추가합니다.
        
        매개변수:
            callback (function): ECG 데이터가 생성될 때마다 호출될 콜백 함수
        """
        if callback not in self.listeners:
            self.listeners.append(callback)
            logger.info("ECG 데이터 리스너가 추가되었습니다.")
    
    def remove_listener(self, callback):
        """
        ECG 데이터 리스너를 제거합니다.
        
        매개변수:
            callback (function): 제거할 콜백 함수
        """
        if callback in self.listeners:
            self.listeners.remove(callback)
            logger.info("ECG 데이터 리스너가 제거되었습니다.")
    
    def set_pattern(self, pattern_type, heart_rate=None, hrv_level=None):
        """
        ECG 패턴을 설정합니다.
        
        매개변수:
            pattern_type (ECGPatternType): 패턴 유형
            heart_rate (int, optional): 심박수 (bpm)
            hrv_level (float, optional): 심박변이도 수준 (0-1)
        """
        if isinstance(pattern_type, str):
            try:
                pattern_type = ECGPatternType(pattern_type)
            except ValueError:
                logger.error(f"유효하지 않은 패턴 유형: {pattern_type}")
                return
        
        self.current_pattern = pattern_type
        
        # 패턴에 따른 기본값 설정
        if heart_rate is None:
            if pattern_type == ECGPatternType.NORMAL:
                heart_rate = 70
            elif pattern_type == ECGPatternType.ELEVATED_HR:
                heart_rate = 100
            elif pattern_type == ECGPatternType.ANXIETY:
                heart_rate = 95
            elif pattern_type == ECGPatternType.ARRHYTHMIA:
                heart_rate = 80
            elif pattern_type == ECGPatternType.REDUCED_HRV:
                heart_rate = 75
        
        if hrv_level is None:
            if pattern_type == ECGPatternType.NORMAL:
                hrv_level = 0.1
            elif pattern_type == ECGPatternType.ELEVATED_HR:
                hrv_level = 0.08
            elif pattern_type == ECGPatternType.ANXIETY:
                hrv_level = 0.05
            elif pattern_type == ECGPatternType.ARRHYTHMIA:
                hrv_level = 0.25
            elif pattern_type == ECGPatternType.REDUCED_HRV:
                hrv_level = 0.02
        
        self.current_heart_rate = heart_rate
        self.heart_rate_variability = hrv_level
        
        logger.info(f"ECG 패턴이 설정되었습니다: {pattern_type.value}, HR={heart_rate}bpm, HRV={hrv_level}")
    
    def start(self):
        """
        ECG 시뮬레이션을 시작합니다.
        """
        if self.running:
            logger.warning("시뮬레이션이 이미 실행 중입니다.")
            return
        
        self.running = True
        self.generated_data = []
        self.timestamps = []
        
        self.simulation_thread = threading.Thread(target=self._simulation_worker)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("ECG 시뮬레이션이 시작되었습니다.")
    
    def stop(self):
        """
        ECG 시뮬레이션을 중지합니다.
        """
        if not self.running:
            logger.warning("시뮬레이션이 실행 중이지 않습니다.")
            return
        
        self.running = False
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        
        logger.info("ECG 시뮬레이션이 중지되었습니다.")
    
    def _simulation_worker(self):
        """
        ECG 시뮬레이션 워커 함수
        """
        start_time = time.time()
        
        # 시뮬레이션 루프
        while self.running:
            try:
                # 현재 패턴 템플릿 가져오기
                template = self.templates[self.current_pattern]
                
                # 현재 심박수에 기반한 RR 간격 계산 (초)
                # 60초 / 심박수 = RR 간격(초)
                rr_interval = 60.0 / self.current_heart_rate
                
                # 현재 패턴에 맞는 HRV 적용
                if self.current_pattern == ECGPatternType.ARRHYTHMIA:
                    # 부정맥은 불규칙한 간격
                    rr_variation = rr_interval * self.heart_rate_variability * np.random.randn() * 2
                else:
                    # 정상 HRV 변이
                    rr_variation = rr_interval * self.heart_rate_variability * np.random.randn()
                
                # RR 간격 변이 적용
                current_rr = max(0.2, min(2.0, rr_interval + rr_variation))  # 0.2-2.0초 범위로 제한
                
                # 템플릿 길이에 맞게 신호 샘플 수 계산
                n_samples = int(current_rr * self.sampling_rate)
                
                # 템플릿보다 긴 경우, 나머지 부분을 0으로 채움
                if n_samples > len(template):
                    ecg_signal = np.zeros(n_samples)
                    ecg_signal[:len(template)] = template
                else:
                    # 템플릿보다 짧은 경우, 템플릿을 자름
                    ecg_signal = template[:n_samples]
                
                # 노이즈 추가
                noise = self.noise_level * np.random.randn(len(ecg_signal))
                ecg_signal = ecg_signal + noise
                
                # 시간 데이터
                current_time = time.time() - start_time
                timestamps = np.linspace(
                    current_time,
                    current_time + current_rr,
                    len(ecg_signal)
                )
                
                # 결과 저장
                self.generated_data.extend(ecg_signal)
                self.timestamps.extend(timestamps)
                
                # 버퍼 크기 제한 (최대 1분)
                max_buffer = 60 * self.sampling_rate
                if len(self.generated_data) > max_buffer:
                    self.generated_data = self.generated_data[-max_buffer:]
                    self.timestamps = self.timestamps[-max_buffer:]
                
                # 리스너에게 알림
                for listener in self.listeners:
                    try:
                        listener({
                            "data": ecg_signal.tolist(),
                            "timestamp": current_time,
                            "sampling_rate": self.sampling_rate,
                            "pattern": self.current_pattern.value,
                            "heart_rate": self.current_heart_rate
                        })
                    except Exception as e:
                        logger.error(f"리스너 호출 중 오류 발생: {e}")
                
                # 다음 심박까지 대기
                time.sleep(current_rr * 0.5)  # 실제 시간의 2배 속도로 시뮬레이션
                
            except Exception as e:
                logger.error(f"ECG 시뮬레이션 중 오류 발생: {e}")
                time.sleep(0.1)
    
    def get_data(self, duration=None):
        """
        생성된 ECG 데이터를 가져옵니다.
        
        매개변수:
            duration (float, optional): 가져올 데이터의 지속 시간 (초). None이면 모든 데이터.
            
        반환값:
            tuple: (ECG 데이터, 타임스탬프)
        """
        if not self.generated_data:
            return np.array([]), np.array([])
        
        if duration is None:
            return np.array(self.generated_data), np.array(self.timestamps)
        
        # 최근 duration 초 데이터만 가져오기
        current_time = self.timestamps[-1]
        start_time = current_time - duration
        
        # 시작 인덱스 찾기
        start_idx = 0
        for i, t in enumerate(self.timestamps):
            if t >= start_time:
                start_idx = i
                break
        
        return np.array(self.generated_data[start_idx:]), np.array(self.timestamps[start_idx:])
    
    def visualize(self, duration=10):
        """
        생성된 ECG 데이터를 시각화합니다.
        
        매개변수:
            duration (float): 시각화할 데이터의 지속 시간 (초)
        """
        data, timestamps = self.get_data(duration)
        
        if len(data) == 0:
            logger.warning("시각화할 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 상대적인 시간으로 변환 (첫 번째 샘플을 0초로)
        rel_timestamps = timestamps - timestamps[0]
        
        plt.plot(rel_timestamps, data)
        plt.title(f"ECG 시뮬레이션 - {self.current_pattern.value}")
        plt.xlabel("시간 (초)")
        plt.ylabel("진폭")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_data(self, filename):
        """
        생성된 ECG 데이터를 파일에 저장합니다.
        
        매개변수:
            filename (str): 저장할 파일 이름
        """
        data, timestamps = self.get_data()
        
        if len(data) == 0:
            logger.warning("저장할 데이터가 없습니다.")
            return
        
        try:
            np.savez(
                filename,
                ecg_data=data,
                timestamps=timestamps,
                sampling_rate=self.sampling_rate,
                pattern=self.current_pattern.value,
                heart_rate=self.current_heart_rate,
                hrv_level=self.heart_rate_variability
            )
            
            logger.info(f"ECG 데이터가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {e}")


# 테스트 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    simulator = ECGSimulator(sampling_rate=256, noise_level=0.05)
    
    def data_callback(data):
        logger.info(f"ECG 데이터 생성: {len(data['data'])} 샘플")
    
    simulator.add_listener(data_callback)
    
    # 정상 패턴으로 시작
    simulator.set_pattern(ECGPatternType.NORMAL)
    simulator.start()
    
    # 5초간 실행 후 불안 패턴으로 변경
    time.sleep(5)
    simulator.set_pattern(ECGPatternType.ANXIETY)
    
    # 추가 5초 실행 후 중지
    time.sleep(5)
    simulator.stop()
    
    # 데이터 시각화
    simulator.visualize(duration=10)
