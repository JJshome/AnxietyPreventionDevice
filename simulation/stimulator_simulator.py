#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
저주파 자극기 시뮬레이터

실제 저주파 자극기 없이 자극 파형과 효과를 시뮬레이트하는 모듈입니다.
특히 두 개 이상의 자극기 간 위상차 설정에 따른 효과를 시뮬레이트합니다.
"""

import numpy as np
import time
import logging
import threading
from queue import Queue

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StimulatorSimulator")


class StimulatorSimulator:
    """
    저주파 자극기 시뮬레이터 클래스
    """
    
    # 파형 유형 상수
    WAVE_SINE = "sine"
    WAVE_SQUARE = "square"
    WAVE_TRIANGLE = "triangle"
    WAVE_BIPHASIC = "biphasic"
    
    def __init__(self, sampling_rate=1000):
        """
        초기화
        
        Args:
            sampling_rate (int): 샘플링 속도 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.stimulators = {}  # 자극기 정보 저장
        
        # 자극 파라미터
        self.waveform = self.WAVE_SINE
        self.frequency = 15.0  # Hz
        self.amplitude = 0.5   # 0~1 값 (최대값의 비율)
        self.phase_difference = 0.3  # 자극기 간 위상차 (0~1, 1은 360도)
        
        # 시뮬레이션 실행 상태
        self.running = False
        self.simulation_thread = None
        
        # 시뮬레이션 데이터 처리
        self.output_queue = Queue(maxsize=100)  # 시뮬레이션 결과 저장
        
        # 자극 효과 파라미터
        self.effect_delay = 2.0  # 자극 효과 나타나기까지 지연시간 (초)
        self.effect_duration = 10.0  # 자극 효과 지속시간 (초)
        self.effect_strength = 0.7  # 자극 효과 강도 (0~1)
        
        # 콜백 함수
        self.callback = None
        
        logger.info(f"저주파 자극기 시뮬레이터 초기화 (샘플링 속도: {sampling_rate}Hz)")
    
    def add_stimulator(self, stim_id, name=None, battery_level=100):
        """
        새로운 자극기 추가
        
        Args:
            stim_id (str): 자극기 ID
            name (str): 자극기 이름
            battery_level (int): 배터리 잔량 (%)
            
        Returns:
            bool: 추가 성공 여부
        """
        if stim_id in self.stimulators:
            logger.warning(f"이미 존재하는 자극기 ID입니다: {stim_id}")
            return False
        
        # 자극기 추가
        if name is None:
            name = f"Stimulator {len(self.stimulators) + 1}"
        
        # 자극기 정보 저장
        self.stimulators[stim_id] = {
            "id": stim_id,
            "name": name,
            "battery_level": battery_level,
            "waveform": self.waveform,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase_offset": 0.0 if len(self.stimulators) == 0 else self.phase_difference,
            "active": False,
            "last_update": time.time()
        }
        
        logger.info(f"자극기 추가됨: {name} (ID: {stim_id})")
        return True
    
    def remove_stimulator(self, stim_id):
        """
        자극기 제거
        
        Args:
            stim_id (str): 자극기 ID
            
        Returns:
            bool: 제거 성공 여부
        """
        if stim_id not in self.stimulators:
            logger.warning(f"존재하지 않는 자극기 ID입니다: {stim_id}")
            return False
        
        # 자극기 삭제
        name = self.stimulators[stim_id]["name"]
        del self.stimulators[stim_id]
        
        logger.info(f"자극기 제거됨: {name} (ID: {stim_id})")
        return True
    
    def start_simulation(self):
        """
        자극기 시뮬레이션 시작
        
        Returns:
            bool: 시작 성공 여부
        """
        if self.running:
            logger.warning("이미 시뮬레이션이 실행 중입니다.")
            return True
        
        # 큐 초기화
        with self.output_queue.mutex:
            self.output_queue.queue.clear()
        
        # 시뮬레이션 스레드 시작
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_worker)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info("저주파 자극기 시뮬레이션 시작")
        return True
    
    def stop_simulation(self):
        """
        자극기 시뮬레이션 중지
        
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
        
        # 모든 자극기 정지
        for stim_id in self.stimulators:
            self.stimulators[stim_id]["active"] = False
        
        logger.info("저주파 자극기 시뮬레이션 중지")
        return True
    
    def set_callback(self, callback_function):
        """
        실시간 시뮬레이션 결과를 받을 콜백 함수 설정
        
        Args:
            callback_function (function): 콜백 함수
        """
        self.callback = callback_function
        logger.info("콜백 함수가 설정되었습니다.")
    
    def start_stimulation(self, params=None):
        """
        자극 시작
        
        Args:
            params (dict): 자극 파라미터 (주파수, 진폭, 파형 등)
            
        Returns:
            bool: 시작 성공 여부
        """
        if not self.stimulators:
            logger.warning("연결된 자극기가 없습니다.")
            return False
        
        # 파라미터 처리
        if params is not None:
            if "frequency" in params:
                self.frequency = max(1.0, min(100.0, params["frequency"]))  # 1~100Hz 범위
            if "amplitude" in params:
                self.amplitude = max(0.0, min(1.0, params["amplitude"]))    # 0~1 범위
            if "waveform" in params and params["waveform"] in [self.WAVE_SINE, self.WAVE_SQUARE, self.WAVE_TRIANGLE, self.WAVE_BIPHASIC]:
                self.waveform = params["waveform"]
            if "phase_difference" in params:
                self.phase_difference = max(0.0, min(1.0, params["phase_difference"]))  # 0~1 범위
            if "effect_delay" in params:
                self.effect_delay = max(0.0, params["effect_delay"])  # 자극 효과 지연시간
            if "effect_duration" in params:
                self.effect_duration = max(0.0, params["effect_duration"])  # 자극 효과 지속시간
            if "effect_strength" in params:
                self.effect_strength = max(0.0, min(1.0, params["effect_strength"]))  # 자극 효과 강도
        
        # 모든 자극기 설정 업데이트
        stimulator_ids = list(self.stimulators.keys())
        for i, stim_id in enumerate(stimulator_ids):
            self.stimulators[stim_id]["active"] = True
            self.stimulators[stim_id]["waveform"] = self.waveform
            self.stimulators[stim_id]["frequency"] = self.frequency
            self.stimulators[stim_id]["amplitude"] = self.amplitude
            
            # 첫 번째 자극기는 기준 위상, 나머지는 위상차 적용
            self.stimulators[stim_id]["phase_offset"] = 0.0 if i == 0 else self.phase_difference
            
            # 배터리 사용량 시뮬레이션
            self.stimulators[stim_id]["battery_level"] = max(0, self.stimulators[stim_id]["battery_level"] - 0.5)
            
            self.stimulators[stim_id]["last_update"] = time.time()
        
        logger.info(f"자극 시작 - 주파수: {self.frequency}Hz, " +
                 f"진폭: {self.amplitude}, " +
                 f"파형: {self.waveform}, " +
                 f"위상차: {self.phase_difference}")
        return True
    
    def stop_stimulation(self):
        """
        자극 중지
        
        Returns:
            bool: 중지 성공 여부
        """
        # 모든 자극기 정지
        for stim_id in self.stimulators:
            self.stimulators[stim_id]["active"] = False
        
        logger.info("자극 중지")
        return True
    
    def set_stimulator_params(self, stim_id, params):
        """
        특정 자극기의 파라미터 설정
        
        Args:
            stim_id (str): 자극기 ID
            params (dict): 파라미터 사전
            
        Returns:
            bool: 설정 성공 여부
        """
        if stim_id not in self.stimulators:
            logger.warning(f"존재하지 않는 자극기 ID입니다: {stim_id}")
            return False
        
        # 파라미터 업데이트
        stim = self.stimulators[stim_id]
        
        if "waveform" in params:
            if params["waveform"] in [self.WAVE_SINE, self.WAVE_SQUARE, self.WAVE_TRIANGLE, self.WAVE_BIPHASIC]:
                stim["waveform"] = params["waveform"]
        
        if "frequency" in params:
            stim["frequency"] = max(1.0, min(100.0, params["frequency"]))  # 1~100Hz 범위
        
        if "amplitude" in params:
            stim["amplitude"] = max(0.0, min(1.0, params["amplitude"]))  # 0~1 범위
        
        if "phase_offset" in params:
            stim["phase_offset"] = max(0.0, min(1.0, params["phase_offset"]))  # 0~1 범위
        
        if "active" in params:
            stim["active"] = bool(params["active"])
        
        stim["last_update"] = time.time()
        
        logger.info(f"자극기 {stim['name']} 파라미터 업데이트")
        return True
    
    def get_battery_level(self, stim_id):
        """
        특정 자극기의 배터리 잔량 조회
        
        Args:
            stim_id (str): 자극기 ID
            
        Returns:
            int: 배터리 잔량 (%)
        """
        if stim_id not in self.stimulators:
            logger.warning(f"존재하지 않는 자극기 ID입니다: {stim_id}")
            return 0
        
        # 배터리 사용량 시뮬레이션 (활성화된 자극기일 경우 더 빠르게 배터리 소모)
        stim = self.stimulators[stim_id]
        current_level = stim["battery_level"]
        
        # 활성화된 자극기는 배터리가 더 빠르게 소모됨
        if stim["active"]:
            new_level = max(0, current_level - np.random.uniform(0, 0.2))  # 최대 0.2% 감소
        else:
            new_level = max(0, current_level - np.random.uniform(0, 0.01))  # 최대 0.01% 감소 (대기상태)
        
        stim["battery_level"] = new_level
        return int(new_level)
    
    def _simulation_worker(self):
        """
        시뮬레이션 스레드 함수
        """
        logger.info("자극기 시뮬레이션 스레드 시작")
        
        # 시뮬레이션 간격
        update_interval = 0.1  # 100ms
        t = 0.0  # 시뮬레이션 시간 (초)
        
        # 시뮬레이션 시작 시간 기록
        stimulation_start_time = 0
        stimulation_active = False
        
        while self.running:
            try:
                # 활성화된 자극기 확인
                active_stimulators = []
                for stim_id, stim in self.stimulators.items():
                    if stim["active"]:
                        active_stimulators.append(stim_id)
                
                # 현재 자극 상태 갱신
                if active_stimulators and not stimulation_active:
                    # 자극 시작
                    stimulation_start_time = t
                    stimulation_active = True
                    logger.info(f"자극 활성화 감지 (시간: {t:.1f}s)")
                elif not active_stimulators and stimulation_active:
                    # 자극 종료
                    stimulation_active = False
                    logger.info(f"자극 비활성화 감지 (시간: {t:.1f}s)")
                
                # 자극 출력 과 효과 시뮬레이션
                if stimulation_active:
                    # 자극 시간 계산
                    stim_time = t - stimulation_start_time
                    
                    # 자극 파형 생성
                    waveforms = {}
                    for stim_id in active_stimulators:
                        stim = self.stimulators[stim_id]
                        wave = self._generate_waveform(
                            waveform=stim["waveform"],
                            frequency=stim["frequency"],
                            amplitude=stim["amplitude"],
                            phase_offset=stim["phase_offset"],
                            duration=update_interval
                        )
                        waveforms[stim_id] = wave
                    
                    # 자극 효과 참조용 정보
                    effect_info = {
                        "stimulation_time": stim_time,
                        "waveforms": waveforms,
                        "effect_delay": self.effect_delay,
                        "effect_duration": self.effect_duration,
                        "effect_strength": self.effect_strength
                    }
                    
                    # 출력 큐에 추가
                    try:
                        self.output_queue.put(effect_info, block=False)
                    except Queue.Full:
                        # 큐가 가득찬 경우 가장 오래된 항목 제거 후 추가
                        self.output_queue.get()
                        self.output_queue.put(effect_info, block=False)
                    
                    # 콜백 호출
                    if self.callback is not None:
                        self.callback(effect_info)
                
                # 시간 증가
                t += update_interval
                
                # 실제 시간 지연
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"자극기 시뮬레이션 오류: {e}")
                time.sleep(0.5)  # 오류 발생 시 지연
        
        logger.info("자극기 시뮬레이션 스레드 종료")
    
    def _generate_waveform(self, waveform, frequency, amplitude, phase_offset=0.0, duration=1.0):
        """
        주어진 파라미터로 파형 생성
        
        Args:
            waveform (str): 파형 유형 (WAVE_SINE, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_BIPHASIC)
            frequency (float): 주파수 (Hz)
            amplitude (float): 진폭 (0~1)
            phase_offset (float): 위상차 (0~1, 1은 360도)
            duration (float): 생성할 파형의 길이 (초)
            
        Returns:
            numpy.ndarray: 생성된 파형 배열
        """
        # 시간 배열 생성
        t = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # 파형 가중치 경사호 계산
        phase_rad = 2 * np.pi * phase_offset  # 0~1 범위를 0~2π 라디안으로 변환
        
        # 파형 유형에 따른 신호 생성
        if waveform == self.WAVE_SINE:
            waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)
        elif waveform == self.WAVE_SQUARE:
            waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase_rad))
        elif waveform == self.WAVE_TRIANGLE:
            waveform = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t + phase_rad))
        elif waveform == self.WAVE_BIPHASIC:
            # 양방향 파형 (양의 파터스트 후 음의 파터스트)
            sine_wave = np.sin(2 * np.pi * frequency * t + phase_rad)
            # 양의 반과 음의 반으로 나누어 음의 반을 뒤집음
            waveform = amplitude * np.where(sine_wave >= 0, sine_wave, -0.5 * sine_wave)
        else:
            # 기본값으로 사인파 사용
            waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)
        
        return waveform
    
    def get_stimulation_effect(self, stim_time):
        """
        자극 시작 후 특정 시간에서의 효과 계산
        
        Args:
            stim_time (float): 자극 시작 후 경과 시간 (초)
            
        Returns:
            float: 효과 강도 (0~1)
        """
        # 자극 효과는 일정 시간(초) 후부터 나타남
        if stim_time < self.effect_delay:
            return 0.0
        
        # 효과 지속 시간 계산
        effect_time = stim_time - self.effect_delay
        if effect_time > self.effect_duration:
            # 효과 지속 시간을 초과한 경우 감소
            decay_time = effect_time - self.effect_duration
            decay_factor = max(0.0, 1.0 - (decay_time / (self.effect_duration * 0.5)))  # 지속시간의 50% 동안 선형적으로 감소
            return self.effect_strength * decay_factor
        else:
            # 효과 지속 시간 내에는 점차 증가
            ramp_factor = min(1.0, effect_time / (self.effect_duration * 0.2))  # 지속시간의 20% 동안 점차 증가
            return self.effect_strength * ramp_factor
    
    def get_simulation_data(self):
        """
        시뮬레이션 결과 데이터 가져오기
        
        Returns:
            dict: 시뮬레이션 결과 데이터
        """
        try:
            return self.output_queue.get(block=False)
        except Queue.Empty:
            return None
        except Exception as e:
            logger.error(f"시뮬레이션 데이터 가져오기 오류: {e}")
            return None
            
    def get_all_stimulators(self):
        """
        모든 자극기 정보 가져오기
        
        Returns:
            dict: 자극기 정보 사전
        """
        # 배터리 정보 갱신
        for stim_id in self.stimulators:
            self.get_battery_level(stim_id)
        
        return self.stimulators.copy()