#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
저주파 자극기 제어 인터페이스 모듈

저주파 자극기와의 통신 및 제어를 위한 인터페이스 코드입니다.
불안장애 예방을 위한 두개전기자극 신호를 발생시키며,
두 개 이상의 자극기를 위상차를 가지고 제어합니다.
"""

import time
import threading
import logging
import math
import numpy as np
from queue import Queue

# 블루투스 통신을 위한 라이브러리 임포트
try:
    import bluetooth
    HAVE_BLUETOOTH = True
except ImportError:
    HAVE_BLUETOOTH = False
    print("[!] PyBluez 라이브러리가 설치되지 않았습니다. 가상 자극기만 사용 가능합니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StimulatorInterface")


class StimulatorInterface:
    """
    저주파 자극기 제어 인터페이스 클래스
    """
    
    # 파형 유형 상수
    WAVE_SINE = "sine"
    WAVE_SQUARE = "square"
    WAVE_TRIANGLE = "triangle"
    WAVE_BIPHASIC = "biphasic"
    
    def __init__(self, use_virtual_device=False):
        """
        초기화
        
        Args:
            use_virtual_device (bool): 가상 자극기 사용 여부
        """
        self.use_virtual_device = use_virtual_device
        self.stimulators = {}  # MAC 주소를 키로 하는 자극기 사전
        self.sockets = {}      # MAC 주소를 키로 하는 소켓 사전
        
        # 현재 실행 상태
        self.running = False
        self.stimulation_thread = None
        
        # 자극 관련 설정
        self.waveform = self.WAVE_SINE
        self.frequency = 15.0  # Hz
        self.amplitude = 0.5   # 0~1 값 (자극기 최대값의 비율)
        self.phase_difference = 0.3  # 자극기 간 위상차 (0~1, 1은 360도)
        self.delay_time = 0.3  # 자극기 간 시간차 (초)
        
        # 각 자극기별 설정
        self.stimulator_settings = {}
        
        # 실시간 상태 갱신을 위한 관리 스레드
        self.monitor_thread = None
        self.monitoring = False
        
        # 콜백 함수
        self.callback = None
        
        logger.info("저주파 자극기 제어 인터페이스 초기화")
    
    def scan_devices(self, timeout=5):
        """
        주변 블루투스 저주파 자극기 검색
        
        Returns:
            list: 검색된 자극기 목록
        """
        if self.use_virtual_device:
            # 가상 자극기 사용 시 2개의 가상 기기 리턴
            return [
                {
                    "name": "Virtual Stimulator 1",
                    "address": "11:22:33:44:55:66",
                    "paired": True,
                    "battery": 90,
                    "type": "MyBeat TENS"
                },
                {
                    "name": "Virtual Stimulator 2",
                    "address": "22:33:44:55:66:77",
                    "paired": True,
                    "battery": 85,
                    "type": "MyBeat TENS"
                }
            ]
        
        # 실제 기기 검색 코드...
        return []
    
    def connect(self, device_address):
        """
        저주파 자극기에 연결
        
        Returns:
            bool: 연결 성공 여부
        """
        if self.use_virtual_device:
            # 가상 자극기 연결
            stimulator_info = {
                "name": f"Virtual Stimulator {'1' if '11:22' in device_address else '2'}",
                "address": device_address,
                "connected": True,
                "battery": 90 if '11:22' in device_address else 85,
                "last_update": time.time(),
                "amplitude": 0.5,  # 기본 자극 강도 (0~1)
                "frequency": 15.0,  # 기본 주파수 (Hz)
                "waveform": self.WAVE_SINE,  # 기본 파형
                "phase_offset": 0.0 if '11:22' in device_address else self.phase_difference  # 위상차
            }
            
            self.stimulators[device_address] = stimulator_info
            self.stimulator_settings[device_address] = {
                "amplitude": 0.5,
                "frequency": 15.0,
                "waveform": self.WAVE_SINE,
                "phase_offset": 0.0 if '11:22' in device_address else self.phase_difference,
                "enabled": True
            }
            
            logger.info(f"가상 자극기에 연결되었습니다: {stimulator_info['name']}")
            return True
        
        # 실제 자극기 연결 코드...
        return False
    
    def disconnect(self, device_address):
        """
        저주파 자극기와 연결 해제
        
        Returns:
            bool: 연결 해제 성공 여부
        """
        if device_address not in self.stimulators:
            return True
        
        # 자극 중지
        if self.running and device_address in self.stimulator_settings:
            self.stop_stimulation()
        
        if self.use_virtual_device:
            # 가상 자극기 연결 해제
            name = self.stimulators[device_address]["name"]
            del self.stimulators[device_address]
            if device_address in self.stimulator_settings:
                del self.stimulator_settings[device_address]
            
            logger.info(f"가상 자극기 연결이 해제되었습니다: {name}")
            return True
        
        # 실제 자극기 연결 해제 코드...
        return True
    
    def disconnect_all(self):
        """
        모든 자극기 연결 해제
        
        Returns:
            bool: 연결 해제 성공 여부
        """
        # 자극 중지
        if self.running:
            self.stop_stimulation()
        
        # 자극기 연결 해제
        success = True
        for address in list(self.stimulators.keys()):
            if not self.disconnect(address):
                success = False
        
        return success
    
    def get_battery_level(self, device_address):
        """
        배터리 잔량 조회
        
        Returns:
            int: 배터리 잔량 (%)
        """
        if device_address not in self.stimulators:
            return 0
        
        if self.use_virtual_device:
            # 가상 자극기 배터리 레벨 감소
            current_level = self.stimulators[device_address]["battery"]
            new_level = max(0, current_level - np.random.uniform(0, 0.2))
            self.stimulators[device_address]["battery"] = new_level
            return int(new_level)
        
        # 실제 배터리 레벨 가져오기 코드...
        return self.stimulators[device_address]["battery"]
    
    def start_stimulation(self, frequency=None, amplitude=None, waveform=None, phase_difference=None, delay_time=None):
        """
        저주파 자극 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        if not self.stimulators:
            logger.error("연결된 자극기가 없습니다.")
            return False
        
        if self.running:
            logger.warning("이미 자극이 실행 중입니다.")
            return True
        
        # 파라미터 설정
        if frequency is not None:
            self.frequency = max(1.0, min(100.0, frequency))  # 1~100Hz 범위 제한
        if amplitude is not None:
            self.amplitude = max(0.0, min(1.0, amplitude))    # 0~1 범위 제한
        if waveform is not None and waveform in [self.WAVE_SINE, self.WAVE_SQUARE, self.WAVE_TRIANGLE, self.WAVE_BIPHASIC]:
            self.waveform = waveform
        if phase_difference is not None:
            self.phase_difference = max(0.0, min(1.0, phase_difference))  # 0~1 범위 제한
        if delay_time is not None:
            self.delay_time = max(0.0, min(2.0, delay_time))  # 0~2초 범위 제한
        
        # 기본 설정 초기화
        addresses = list(self.stimulators.keys())
        for i, address in enumerate(addresses):
            # 첫 번째 자극기는 기본 설정, 나머지는 위상차 적용
            self.stimulator_settings[address] = {
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "waveform": self.waveform,
                "phase_offset": 0.0 if i == 0 else self.phase_difference,
                "enabled": True
            }
        
        # 자극 스레드 시작
        self.running = True
        self.stimulation_thread = threading.Thread(target=self._stimulation_worker)
        self.stimulation_thread.daemon = True
        self.stimulation_thread.start()
        
        logger.info(f"저주파 자극 시작 - 주파수: {self.frequency}Hz, " +
                 f"진폭: {self.amplitude}, " +
                 f"파형: {self.waveform}, " +
                 f"위상차: {self.phase_difference}, " +
                 f"지연시간: {self.delay_time}s")
        return True
    
    def stop_stimulation(self):
        """
        저주파 자극 중지
        
        Returns:
            bool: 중지 성공 여부
        """
        if not self.running:
            return True
        
        # 자극 스레드 중지
        self.running = False
        if self.stimulation_thread and self.stimulation_thread.is_alive():
            self.stimulation_thread.join(timeout=2.0)
        self.stimulation_thread = None
        
        logger.info("저주파 자극 중지")
        return True
    
    def _stimulation_worker(self):
        """
        자극 작업자 스레드 함수
        """
        logger.info("자극 스레드 시작")
        
        # 가상 자극 모드에서는 실제 신호를 생성하지 않고 시뮬레이션만 수행
        if self.use_virtual_device:
            while self.running:
                # 콜백을 통한 상태 업데이트
                if self.callback is not None:
                    try:
                        state = {
                            'timestamp': time.time(),
                            'stimulators': {addr: self.stimulator_settings[addr].copy() for addr in self.stimulator_settings}
                        }
                        self.callback(state)
                    except Exception as e:
                        logger.error(f"콜백 함수 오류: {e}")
                
                time.sleep(0.1)
        
        logger.info("자극 스레드 종료")
    
    def set_stimulator_params(self, device_address, amplitude=None, frequency=None, waveform=None, phase_offset=None, enabled=None):
        """
        특정 자극기의 파라미터 설정
        
        Returns:
            bool: 설정 성공 여부
        """
        if device_address not in self.stimulators:
            return False
        
        if device_address not in self.stimulator_settings:
            # 설정 초기화
            self.stimulator_settings[device_address] = {
                "amplitude": 0.5,
                "frequency": 15.0,
                "waveform": self.WAVE_SINE,
                "phase_offset": 0.0,
                "enabled": True
            }
        
        # 파라미터 업데이트
        settings = self.stimulator_settings[device_address]
        if amplitude is not None:
            settings["amplitude"] = max(0.0, min(1.0, amplitude))
        if frequency is not None:
            settings["frequency"] = max(1.0, min(100.0, frequency))
        if waveform is not None and waveform in [self.WAVE_SINE, self.WAVE_SQUARE, self.WAVE_TRIANGLE, self.WAVE_BIPHASIC]:
            settings["waveform"] = waveform
        if phase_offset is not None:
            settings["phase_offset"] = max(0.0, min(1.0, phase_offset))
        if enabled is not None:
            settings["enabled"] = bool(enabled)
        
        return True
    
    def set_callback(self, callback_function):
        """
        자극 상태 변경 시 알림을 받을 콜백 함수 설정
        
        Args:
            callback_function (function): 콜백 함수
        """
        self.callback = callback_function
        logger.info("콜백 함수가 설정되었습니다.")
    
    def set_phase_difference(self, phase_difference):
        """
        자극기 간 위상차 설정
        
        Args:
            phase_difference (float): 위상차 (0~1)
            
        Returns:
            bool: 설정 성공 여부
        """
        if phase_difference < 0 or phase_difference > 1:
            return False
        
        self.phase_difference = phase_difference
        
        # 현재 연결된 자극기에 적용
        addresses = list(self.stimulators.keys())
        if len(addresses) >= 2:
            for i, address in enumerate(addresses[1:], 1):
                if address in self.stimulator_settings:
                    self.stimulator_settings[address]["phase_offset"] = self.phase_difference
        
        return True
    
    def set_delay_time(self, delay_time):
        """
        자극기 간 시간차 설정
        
        Args:
            delay_time (float): 시간차 (초, 0~2)
            
        Returns:
            bool: 설정 성공 여부
        """
        if delay_time < 0 or delay_time > 2:
            return False
        
        self.delay_time = delay_time
        return True
    
    def generate_waveform(self, waveform_type, frequency, amplitude, phase_offset=0.0, duration=1.0, sampling_rate=1000):
        """
        주어진 파라미터로 파형 생성 (시리얼화 및 시각화 목적)
        
        Args:
            waveform_type (str): 파형 유형 (WAVE_SINE, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_BIPHASIC)
            frequency (float): 주파수 (Hz)
            amplitude (float): 진폭 (0~1)
            phase_offset (float): 위상차 (0~1, 1은 360도)
            duration (float): 생성할 파형의 길이 (초)
            sampling_rate (int): 샘플링 속도 (Hz)
            
        Returns:
            numpy.ndarray: 생성된 파형 배열
        """
        # 시간 배열 생성
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # 파형 가중치 경사호 계산
        phase_rad = 2 * np.pi * phase_offset  # 0~1 범위를 0~2π 라디안으로 변환
        
        # 파형 유형에 따른 신호 생성
        if waveform_type == self.WAVE_SINE:
            waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)
        elif waveform_type == self.WAVE_SQUARE:
            waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase_rad))
        elif waveform_type == self.WAVE_TRIANGLE:
            waveform = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t + phase_rad))
        elif waveform_type == self.WAVE_BIPHASIC:
            # 양방향 파형 (양의 파터스트 후 음의 파터스트)
            sine_wave = np.sin(2 * np.pi * frequency * t + phase_rad)
            # 양의 반과 음의 반으로 나누어 음의 반을 뒤집음
            waveform = amplitude * np.where(sine_wave >= 0, sine_wave, -0.5 * sine_wave)
        else:
            # 기본값으로 사인파 사용
            waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)
        
        return waveform
