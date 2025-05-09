"""
스테레오 자극 제어 모듈 (특허 10-2459338 기반)

두 개의 저주파 자극기를 스테레오 방식으로 제어하는 클래스를 구현합니다.
각 자극기는 서로 다른 위상의 신호를 받으며, 위상 차이 조절이 가능합니다.
"""

import numpy as np
import time
import threading
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from .stimulator_interface import StimulatorInterface

# 로깅 설정
logger = logging.getLogger(__name__)

class WaveformType(Enum):
    """자극 파형 타입 열거형"""
    SINE = "sine"
    SQUARE = "square"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    MONOPHASIC = "monophasic"
    BIPHASIC = "biphasic"


class StereoStimulator:
    """
    스테레오 저주파 자극기 제어 클래스
    
    특허 10-2459338에 기반한 구현으로, 두 개 이상의 저주파 자극기에
    위상차를 갖는 신호를 인가하여 스테레오 자극 효과를 제공합니다.
    """

    def __init__(self):
        """스테레오 자극기 제어 클래스 초기화"""
        self.stimulators: Dict[str, StimulatorInterface] = {}
        self.is_active = False
        self.current_config = {}
        self.stimulation_thread = None
        
        # 기본 자극 파라미터 설정
        self.default_params = {
            "frequency": 30.0,       # Hz
            "pulse_width": 100.0,    # μs
            "amplitude": 2.0,        # mA
            "waveform": WaveformType.BIPHASIC,
            "duration": 1200,        # seconds (20 minutes)
            "phase_delay": 0.5,      # seconds
            "ramp_up": 5.0,          # seconds
            "ramp_down": 5.0,        # seconds
        }
        
    def register_stimulator(self, stimulator_id: str, stimulator: StimulatorInterface) -> bool:
        """
        저주파 자극기를 등록합니다.
        
        Args:
            stimulator_id: 자극기 고유 식별자
            stimulator: 자극기 인터페이스 구현 객체
            
        Returns:
            성공 여부
        """
        if stimulator_id in self.stimulators:
            logger.warning(f"Stimulator {stimulator_id} is already registered")
            return False
        
        self.stimulators[stimulator_id] = stimulator
        logger.info(f"Stimulator {stimulator_id} registered successfully")
        return True
        
    def unregister_stimulator(self, stimulator_id: str) -> bool:
        """
        등록된 자극기를 제거합니다.
        
        Args:
            stimulator_id: 자극기 고유 식별자
            
        Returns:
            성공 여부
        """
        if stimulator_id not in self.stimulators:
            logger.warning(f"Stimulator {stimulator_id} not found")
            return False
        
        # 활성 상태라면 자극 중지
        if self.is_active:
            self.stop_stimulation()
            
        del self.stimulators[stimulator_id]
        logger.info(f"Stimulator {stimulator_id} unregistered successfully")
        return True
        
    def get_stimulators(self) -> List[str]:
        """
        등록된 모든 자극기 ID 목록을 반환합니다.
        
        Returns:
            자극기 ID 리스트
        """
        return list(self.stimulators.keys())
        
    def get_stimulator_status(self, stimulator_id: str) -> Dict:
        """
        특정 자극기의 상태를 반환합니다.
        
        Args:
            stimulator_id: 자극기 고유 식별자
            
        Returns:
            자극기 상태 정보 딕셔너리
        """
        if stimulator_id not in self.stimulators:
            logger.warning(f"Stimulator {stimulator_id} not found")
            return {"error": "stimulator_not_found"}
            
        return self.stimulators[stimulator_id].get_status()
        
    def start_stimulation(self, params: Optional[Dict] = None) -> bool:
        """
        스테레오 자극을 시작합니다.
        
        Args:
            params: 자극 파라미터 딕셔너리 (None인 경우 기본값 사용)
            
        Returns:
            성공 여부
        """
        if len(self.stimulators) < 2:
            logger.error("At least two stimulators are required for stereo stimulation")
            return False
            
        if self.is_active:
            logger.warning("Stimulation is already active")
            return False
            
        # 파라미터 설정
        self.current_config = self.default_params.copy()
        if params:
            self.current_config.update(params)
            
        # 스레드 시작
        self.is_active = True
        self.stimulation_thread = threading.Thread(
            target=self._stimulation_worker,
            daemon=True
        )
        self.stimulation_thread.start()
        logger.info("Stereo stimulation started")
        return True
        
    def stop_stimulation(self) -> bool:
        """
        스테레오 자극을 중지합니다.
        
        Returns:
            성공 여부
        """
        if not self.is_active:
            logger.warning("Stimulation is not active")
            return False
            
        self.is_active = False
        if self.stimulation_thread:
            self.stimulation_thread.join(timeout=2.0)
            
        # 모든 자극기 중지
        for stim_id, stimulator in self.stimulators.items():
            stimulator.stop()
            logger.info(f"Stopped stimulator {stim_id}")
            
        logger.info("Stereo stimulation stopped")
        return True
        
    def set_phase_delay(self, delay: float) -> bool:
        """
        자극기 간 위상 지연을 설정합니다.
        
        Args:
            delay: 위상 지연 시간 (초, 0.1-1.0초 범위)
            
        Returns:
            성공 여부
        """
        if delay < 0.1 or delay > 1.0:
            logger.error("Phase delay must be between 0.1 and 1.0 seconds")
            return False
            
        self.current_config["phase_delay"] = delay
        logger.info(f"Phase delay set to {delay} seconds")
        return True
        
    def set_balance(self, balance_params: Dict[str, float]) -> bool:
        """
        각 자극기의 자극 강도 밸런스를 설정합니다.
        
        Args:
            balance_params: {자극기ID: 강도(0.0-1.0)} 형식의 딕셔너리
            
        Returns:
            성공 여부
        """
        for stim_id, level in balance_params.items():
            if stim_id not in self.stimulators:
                logger.error(f"Stimulator {stim_id} not found")
                return False
                
            if level < 0.0 or level > 1.0:
                logger.error(f"Balance level must be between 0.0 and 1.0, got {level}")
                return False
                
        # 현재 설정에 반영
        if "balance" not in self.current_config:
            self.current_config["balance"] = {}
            
        self.current_config["balance"].update(balance_params)
        logger.info(f"Balance updated: {balance_params}")
        return True
        
    def synchronize(self) -> bool:
        """
        모든 자극기와 동기화를 수행합니다.
        
        Returns:
            성공 여부
        """
        success = True
        for stim_id, stimulator in self.stimulators.items():
            if not stimulator.synchronize():
                logger.error(f"Failed to synchronize stimulator {stim_id}")
                success = False
                
        return success
        
    def _generate_waveform(self, t: float, config: Dict) -> float:
        """
        특정 시간(t)에 대한 자극 파형 값을 생성합니다.
        
        Args:
            t: 시간(초)
            config: 자극 설정 딕셔너리
            
        Returns:
            해당 시점의 자극 강도 값
        """
        freq = config["frequency"]
        amp = config["amplitude"]
        phase = 0.0  # 기본 위상
        
        # 파형 타입에 따른 계산
        waveform = config["waveform"]
        
        if waveform == WaveformType.SINE:
            return amp * np.sin(2 * np.pi * freq * t + phase)
        elif waveform == WaveformType.SQUARE:
            return amp * np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif waveform == WaveformType.TRIANGLE:
            return amp * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t + phase))
        elif waveform == WaveformType.SAWTOOTH:
            return amp * (2 * (freq * t + phase / (2 * np.pi) - np.floor(freq * t + phase / (2 * np.pi) + 0.5)))
        elif waveform == WaveformType.MONOPHASIC:
            val = np.sin(2 * np.pi * freq * t + phase)
            return amp * max(0, val)
        elif waveform == WaveformType.BIPHASIC:
            # 직접 임구현한 biphasic 파형
            cycle = (t * freq) % 1.0
            if cycle < 0.2:
                return amp
            elif cycle < 0.3:
                return 0
            elif cycle < 0.5:
                return -amp
            else:
                return 0
        else:
            return 0
            
    def _stimulation_worker(self):
        """자극 생성 및 제어 워커 스레드"""
        logger.info("Stimulation worker thread started")
        
        stimulator_list = list(self.stimulators.keys())
        num_stimulators = len(stimulator_list)
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 각 자극기 초기화
        for stim_id, stimulator in self.stimulators.items():
            stimulator.initialize(self.current_config)
            stimulator.start()
            
        try:
            while self.is_active:
                current_time = time.time() - start_time
                
                # 지정된 자극 시간을 초과한 경우
                if current_time >= self.current_config["duration"]:
                    logger.info("Stimulation duration reached")
                    self.is_active = False
                    break
                    
                # 각 자극기에 신호 전송 (위상 지연 적용)
                for idx, stim_id in enumerate(stimulator_list):
                    # 위상 지연 계산 (초)
                    phase_offset = idx * self.current_config["phase_delay"]
                    
                    # 현재 시간에 위상 지연을 적용
                    adjusted_time = current_time - phase_offset
                    
                    # 음수 시간은 0으로 처리 (시작 전)
                    if adjusted_time < 0:
                        continue
                        
                    # 자극 세기 계산
                    amplitude = self._calculate_amplitude(adjusted_time, stim_id)
                    
                    # 자극 파형 값 계산
                    waveform_value = self._generate_waveform(adjusted_time, self.current_config)
                    
                    # 최종 자극 강도 계산
                    final_intensity = amplitude * waveform_value
                    
                    # 자극기에 신호 전송
                    self.stimulators[stim_id].set_intensity(final_intensity)
                    
                # 50ms 대기 (20Hz 업데이트 속도)
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"Error in stimulation worker: {e}")
        finally:
            # 모든 자극기 중지
            for stim_id, stimulator in self.stimulators.items():
                stimulator.stop()
                
            self.is_active = False
            logger.info("Stimulation worker thread ended")
            
    def _calculate_amplitude(self, t: float, stim_id: str) -> float:
        """
        특정 시간에 대한 자극 강도를 계산합니다.
        램프업/다운 및 밸런스 설정 적용.
        
        Args:
            t: 경과 시간(초)
            stim_id: 자극기 ID
            
        Returns:
            계산된 자극 강도 (0.0-1.0)
        """
        config = self.current_config
        duration = config["duration"]
        ramp_up = config["ramp_up"]
        ramp_down = config["ramp_down"]
        
        # 기본 진폭
        amplitude = 1.0
        
        # 램프업 적용
        if t < ramp_up:
            amplitude = t / ramp_up
            
        # 램프다운 적용
        elif t > (duration - ramp_down):
            amplitude = (duration - t) / ramp_down
            
        # 밸런스 적용
        if "balance" in config and stim_id in config["balance"]:
            amplitude *= config["balance"][stim_id]
            
        return max(0.0, min(1.0, amplitude))  # 0.0-1.0 범위로 클리핑


class MusicDrivenStereoStimulator(StereoStimulator):
    """
    음악 기반 스테레오 자극기 제어 클래스
    
    특허 10-2459338에 기반하여, 음악의 비트와 음정에 따라
    저주파 자극 패턴을 생성하는 확장 클래스입니다.
    """
    
    def __init__(self):
        """음악 기반 자극기 제어 클래스 초기화"""
        super().__init__()
        self.audio_features = {
            "beats": [],       # 비트 타임스탬프 목록
            "beat_strength": [],  # 비트 강도 목록
            "pitch": 0.0,      # 현재 음정
            "energy": 0.0,     # 현재 에너지 레벨
            "tempo": 120.0,    # 템포 (BPM)
        }
        self.audio_timestamp = 0.0  # 오디오 재생 시작 시간
    
    def update_audio_features(self, features: Dict) -> None:
        """
        오디오 특성 업데이트
        
        Args:
            features: 오디오 분석 특성 딕셔너리
        """
        self.audio_features.update(features)
        
        # 음원이 새로 시작된 경우 타임스탬프 리셋
        if "new_track" in features and features["new_track"]:
            self.audio_timestamp = time.time()
            
        logger.debug(f"Audio features updated: tempo={self.audio_features['tempo']} BPM")
    
    def _generate_waveform(self, t: float, config: Dict) -> float:
        """
        음악 특성에 따른 자극 파형 생성
        
        Args:
            t: 시간(초)
            config: 자극 설정 딕셔너리
            
        Returns:
            해당 시점의 자극 강도 값
        """
        if not self.audio_features["beats"]:
            # 비트 정보가 없으면 기본 파형 사용
            return super()._generate_waveform(t, config)
            
        # 오디오 재생 경과 시간
        audio_t = time.time() - self.audio_timestamp
        
        # 현재 시간에 가장 가까운 비트 찾기
        beat_time = 0.0
        beat_strength = 1.0
        
        for i, beat in enumerate(self.audio_features["beats"]):
            if beat > audio_t:
                if i > 0:
                    beat_time = self.audio_features["beats"][i-1]
                    beat_strength = self.audio_features["beat_strength"][i-1]
                break
        
        # 비트로부터의 시간 간격
        time_since_beat = audio_t - beat_time
        
        # 비트 기반 주파수 조정 (템포에 따라)
        base_freq = config["frequency"]
        tempo_factor = self.audio_features["tempo"] / 120.0  # 120 BPM 기준
        freq = base_freq * tempo_factor
        
        # 음정에 따른 진폭 조정
        amp = config["amplitude"] * (0.8 + 0.4 * self.audio_features["energy"])
        
        # 비트 강도에 따른 추가 조정
        amp *= beat_strength
        
        # 비트 효과: 비트마다 강한 시작, 감쇠 효과
        decay = 0.3  # 감쇠 상수
        beat_amp = amp * np.exp(-time_since_beat / decay)
        
        # 기본 파형 계산 (주파수 조정)
        waveform = config["waveform"]
        phase = 0.0
        
        if waveform == WaveformType.SINE:
            return beat_amp * np.sin(2 * np.pi * freq * t + phase)
        elif waveform == WaveformType.BIPHASIC:
            # 음악 비트에 동기화된 biphasic 파형
            cycle = (t * freq) % 1.0
            if cycle < 0.2:
                return beat_amp
            elif cycle < 0.3:
                return 0
            elif cycle < 0.5:
                return -beat_amp
            else:
                return 0
        else:
            # 다른 파형은 기본 구현 사용
            basic_wave = super()._generate_waveform(t, {**config, "amplitude": 1.0, "frequency": freq})
            return beat_amp * basic_wave
