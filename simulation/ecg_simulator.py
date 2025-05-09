#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECG 신호 시뮬레이터

실제 심전도 센서 없이 다양한 ECG 신호를 생성하여 시뮬레이션하는 모듈입니다.
불안장애 시나리오를 시뮬레이트하기 위한 다양한 심박 변화를 생성할 수 있습니다.
"""

import numpy as np
import time
import threading
import logging
from queue import Queue

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ECGSimulator")


class ECGSimulator:
    """
    ECG 신호 시뮬레이터 클래스
    """
    
    # 심장 상태 상수
    STATE_NORMAL = "normal"          # 정상 상태
    STATE_STRESS = "stress"          # 스트레스 상태
    STATE_ANXIETY = "anxiety"        # 불안 상태
    STATE_RELAXED = "relaxed"        # 편안 상태
    
    def __init__(self, sampling_rate=256):
        """
        초기화
        
        Args:
            sampling_rate (int): 샘플링 속도 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.data_queue = Queue(maxsize=100)
        
        # 심장 상태 관련 변수
        self.heart_rate = 60  # 기본 심박수 (bpm)
        self.heart_rate_variability = 5  # 심박 변동성 (정상상태)
        self.current_state = self.STATE_NORMAL
        self.state_transition = None  # 상태 전환 정보 (시작 시간, 지속 시간, 대상 상태)
        
        # 시뮬레이션 실행 상태
        self.running = False
        self.simulation_thread = None
        
        # 콜백 함수
        self.callback = None
        
        logger.info(f"ECG 시뮬레이터 초기화 (샘플링 속도: {sampling_rate}Hz)")
    
    def start_simulation(self):
        """
        ECG 신호 시뮬레이션 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        if self.running:
            logger.warning("이미 시뮬레이션이 실행 중입니다.")
            return True
        
        # 큐 초기화
        with self.data_queue.mutex:
            self.data_queue.queue.clear()
        
        # 식시뮬레이션 스레드 시작
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_worker)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("ECG 신호 시뮬레이션 시작")
        return True
    
    def stop_simulation(self):
        """
        ECG 신호 시뮬레이션 중지
        
        Returns:
            bool: 중지 성공 여부
        """
        if not self.running:
            logger.warning("시뮬레이션이 실행 중이지 않습니다.")
            return True
        
        # 시뮬레이션 스레드 중지
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        self.simulation_thread = None
        
        logger.info("ECG 신호 시뮬레이션 중지")
        return True
    
    def set_callback(self, callback_function):
        """
        새로운 ECG 데이터가 있을 때 호출될 콜백 함수 설정
        
        Args:
            callback_function (function): 콜백 함수
        """
        self.callback = callback_function
        logger.info("콜백 함수가 설정되었습니다.")
    
    def get_data(self, block=True, timeout=1.0):
        """
        시뮬레이션된 ECG 데이터 가져오기
        
        Args:
            block (bool): 데이터가 없을 경우 대기 여부
            timeout (float): 대기 시간(초)
            
        Returns:
            numpy.ndarray: ECG 데이터 또는 None
        """
        if not self.running:
            logger.warning("시뮬레이션이 실행 중이지 않습니다.")
            return None
        
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except Queue.Empty:
            return None
        except Exception as e:
            logger.error(f"데이터 가져오기 오류: {e}")
            return None
    
    def _simulation_worker(self):
        """
        ECG 신호 시뮬레이션 스레드 함수
        """
        logger.info("ECG 시뮬레이션 스레드 시작")
        
        # 시뮬레이션 변수
        chunk_size = int(self.sampling_rate * 0.2)  # 0.2초 단위로 데이터 생성
        t = 0.0  # 총 시뮬레이션 시간 (초)
        
        # 시뮬레이션 루프
        while self.running:
            try:
                # 현재 상태 확인 및 상태 전환 처리
                self._check_state_transition(t)
                
                # 현재 상태에 따른 심박수 및 변동성 조절
                heart_rate, hrv = self._get_heart_parameters()
                
                # ECG 신호 채널 생성
                ecg_signal = self._generate_ecg_signal(chunk_size / self.sampling_rate, heart_rate, hrv)
                
                # 큐에 데이터 추가
                try:
                    self.data_queue.put(ecg_signal, block=False)
                    
                    # 콜백 실행
                    if self.callback is not None:
                        self.callback(ecg_signal)
                        
                except Queue.Full:
                    # 큐가 가득찬 경우 가장 오래된 항목 제거 후 추가
                    self.data_queue.get()
                    self.data_queue.put(ecg_signal, block=False)
                
                # 시간 증가
                t += chunk_size / self.sampling_rate
                
                # 실제 하드웨어 시간 시뮬레이션
                time.sleep(chunk_size / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"ECG 시뮬레이션 오류: {e}")
                time.sleep(0.5)  # 오류 발생 시 잠시 대기
        
        logger.info("ECG 시뮬레이션 스레드 종료")
    
    def _check_state_transition(self, current_time):
        """
        상태 전환 확인 및 처리
        
        Args:
            current_time (float): 현재 시뮬레이션 시간 (초)
        """
        # 상태 전환 정보가 있을 경우 처리
        if self.state_transition is not None:
            start_time, duration, target_state = self.state_transition
            
            # 상태 전환 시작 시간 확인
            if current_time >= start_time and self.current_state != target_state:
                # 상태 전환 시작
                self.current_state = target_state
                logger.info(f"심장 상태 전환: {self.current_state} (시작 시간: {start_time:.1f}s, 지속 시간: {duration:.1f}s)")
            
            # 상태 전환 종료 시간 확인
            if current_time >= start_time + duration and self.current_state == target_state:
                # 상태 전환 종료, 정상 상태로 복귀
                self.current_state = self.STATE_NORMAL
                self.state_transition = None  # 상태 전환 정보 초기화
                logger.info(f"심장 상태 복귀: {self.current_state} (시간: {current_time:.1f}s)")
    
    def _get_heart_parameters(self):
        """
        현재 상태에 따른 심박수 및 HRV 값 가져오기
        
        Returns:
            tuple: (heart_rate, hrv) - 심박수(bpm)와 심박변동성 값
        """
        # 상태별 기본 파라미터
        if self.current_state == self.STATE_NORMAL:
            base_hr = 60 + np.random.uniform(-5, 5)  # 정상 심박수: 55~65 bpm
            base_hrv = 5 + np.random.uniform(-1, 1)  # 정상 HRV: 4~6
        elif self.current_state == self.STATE_STRESS:
            base_hr = 85 + np.random.uniform(-5, 10)  # 스트레스 시 심박수: 80~95 bpm
            base_hrv = 3 + np.random.uniform(-1, 1)   # 스트레스 시 HRV: 2~4 (감소)
        elif self.current_state == self.STATE_ANXIETY:
            base_hr = 100 + np.random.uniform(-5, 15)  # 불안 시 심박수: 95~115 bpm
            base_hrv = 2 + np.random.uniform(-0.5, 0.5) # 불안 시 HRV: 1.5~2.5 (크게 감소)
        elif self.current_state == self.STATE_RELAXED:
            base_hr = 55 + np.random.uniform(-5, 5)    # 편안 시 심박수: 50~60 bpm
            base_hrv = 8 + np.random.uniform(-1, 2)    # 편안 시 HRV: 7~10 (증가)
        else:
            # 기본값 (정상 상태)
            base_hr = 60 + np.random.uniform(-5, 5)
            base_hrv = 5 + np.random.uniform(-1, 1)
        
        # 현재 값 업데이트
        self.heart_rate = base_hr
        self.heart_rate_variability = base_hrv
        
        return base_hr, base_hrv
    
    def _generate_ecg_signal(self, duration, heart_rate, hrv, noise_level=0.05):
        """
        ECG 신호 생성
        
        Args:
            duration (float): 생성할 신호 길이 (초)
            heart_rate (float): 심박수 (bpm)
            hrv (float): 심박 변동성 값
            noise_level (float): 잡음 수준 (0~1)
            
        Returns:
            numpy.ndarray: 생성된 ECG 신호
        """
        # 시간 보독 만들기
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # 기본 심박에 변동성 추가 (상태에 따라 다름)
        period = 60.0 / heart_rate  # 심박수를 주기(초)로 변환
        
        # 심박 변동성 반영
        # HRV가 낮을수록 더 규칙적인 심박, 높을수록 덤 변동성 있는 심박
        hrv_factor = hrv / 10.0  # 0~1 범위로 정규화
        
        # 심장 주기에 변동성 추가
        heart_phase = np.cumsum(1.0 + hrv_factor * np.random.normal(0, 0.1, num_samples)) / self.sampling_rate / period
        heart_phase = heart_phase % 1.0  # 0~1 범위로 정규화
        
        # ECG 파형 생성 (P-QRS-T 파형)
        ecg_signal = self._generate_ecg_waveform(heart_phase)
        
        # 잡음 추가
        if noise_level > 0:
            noise = noise_level * np.random.normal(0, 1, num_samples)
            # 현재 상태에 따라 잡음 정도 조정
            if self.current_state == self.STATE_ANXIETY:
                # 불안 상태에서는 잡음이 더 심함
                noise = noise_level * 1.5 * np.random.normal(0, 1, num_samples)
            ecg_signal += noise
        
        return ecg_signal
    
    def _generate_ecg_waveform(self, heart_phase):
        """
        단일 심장 싸이클의 ECG 파형 생성 (P-QRS-T 파형)
        
        Args:
            heart_phase (numpy.ndarray): 심장 주기 위상 (0~1 범위의 값)
            
        Returns:
            numpy.ndarray: 생성된 ECG 파형
        """
        # 파라미터 설정
        p_time = 0.2    # P-wave 발생 시점 (0~1 사이클 내)
        p_width = 0.08  # P-wave 폭
        p_amp = 0.25    # P-wave 진폭
        
        qrs_time = 0.4  # QRS 발생 시점
        q_width = 0.02  # Q-wave 폭
        q_amp = -0.125   # Q-wave 진폭 (음수값)
        r_width = 0.02  # R-wave 폭
        r_amp = 1.0     # R-wave 진폭 (ECG에서 가장 큰 피크)
        s_width = 0.02  # S-wave 폭
        s_amp = -0.3    # S-wave 진폭 (음수값)
        
        t_time = 0.6   # T-wave 발생 시점
        t_width = 0.1   # T-wave 폭
        t_amp = 0.35    # T-wave 진폭
        
        # 가우시안 함수 (피크 생성용)
        def gaussian(x, amp, mean, width):
            return amp * np.exp(-((x - mean) ** 2) / (2 * width ** 2))
        
        # ECG 파형 생성
        p_wave = gaussian(heart_phase, p_amp, p_time, p_width)
        q_wave = gaussian(heart_phase, q_amp, qrs_time - q_width, q_width)
        r_wave = gaussian(heart_phase, r_amp, qrs_time, r_width)
        s_wave = gaussian(heart_phase, s_amp, qrs_time + s_width, s_width)
        t_wave = gaussian(heart_phase, t_amp, t_time, t_width)
        
        # 모든 파형 합성
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave
        
        # 부정수 조정 (0.1 정도의 기준선)
        baseline = 0.1 * np.sin(2 * np.pi * heart_phase * 0.5)  # 완만한 기준선 엄도 추가
        
        return ecg + baseline
    
    def simulate_stress_event(self, start_time=None, duration=60.0, intensity=0.8):
        """
        스트레스/불안 상태 시뮬레이션
        
        Args:
            start_time (float): 이벤트 시작 시간 (초), None이면 현재 시간
            duration (float): 이벤트 지속 시간 (초)
            intensity (float): 강도 (0~1), 1에 가까울수록 불안 상태
            
        Returns:
            bool: 설정 성공 여부
        """
        if not self.running:
            logger.warning("시뮬레이션이 실행 중이지 않습니다.")
            return False
        
        # 강도에 따라 상태 결정 (스트레스 또는 불안)
        target_state = self.STATE_STRESS if intensity < 0.7 else self.STATE_ANXIETY
        
        # 시작 시간이 None이면 현재 시간 사용
        if start_time is None:
            # 현재 시뮬레이션 시간 추정 (예상)
            if hasattr(self, 'simulation_start_time'):
                current_time = time.time() - self.simulation_start_time
            else:
                current_time = 0
            start_time = current_time
        
        # 상태 전환 정보 설정
        self.state_transition = (start_time, duration, target_state)
        
        logger.info(f"스트레스/불안 이벤트 시뮬레이션 설정 - 상태: {target_state}, " +
                 f"시작 시간: {start_time:.1f}s, 지속 시간: {duration:.1f}s, 강도: {intensity:.2f}")
        return True
    
    def simulate_relaxation(self, start_time=None, duration=30.0, target_level=0.1):
        """
        편안 상태 시뮬레이션 (자극기 사용 후 회복 효과 시뮬레이션)
        
        Args:
            start_time (float): 이벤트 시작 시간 (초), None이면 현재 시간
            duration (float): 이벤트 지속 시간 (초)
            target_level (float): 목표 수치 (0~1), 0에 가까울수록 더 편안
            
        Returns:
            bool: 설정 성공 여부
        """
        if not self.running:
            logger.warning("시뮬레이션이 실행 중이지 않습니다.")
            return False
        
        # 시작 시간이 None이면 현재 시간 사용
        if start_time is None:
            # 현재 시뮬레이션 시간 추정 (예상)
            if hasattr(self, 'simulation_start_time'):
                current_time = time.time() - self.simulation_start_time
            else:
                current_time = 0
            start_time = current_time
        
        # 상태 전환 정보 설정
        self.state_transition = (start_time, duration, self.STATE_RELAXED)
        
        logger.info(f"편안 상태 시뮬레이션 설정 - 시작 시간: {start_time:.1f}s, " +
                 f"지속 시간: {duration:.1f}s, 목표 수치: {target_level:.2f}")
        return True
    
    def get_current_state(self):
        """
        현재 시뮬레이션 상태 가져오기
        
        Returns:
            dict: 현재 시뮬레이션 상태 정보
        """
        return {
            'state': self.current_state,
            'heart_rate': self.heart_rate,
            'heart_rate_variability': self.heart_rate_variability,
            'state_transition': self.state_transition,
            'running': self.running
        }
