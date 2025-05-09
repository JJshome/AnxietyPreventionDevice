import numpy as np
import time
import threading
import logging
import json
import uuid
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class WaveformType(Enum):
    """자극 파형 유형"""
    SINE = "sine"         # 정현파
    SQUARE = "square"     # 구형파
    TRIANGLE = "triangle" # 삼각파
    SAWTOOTH = "sawtooth" # 톱니파
    PULSE = "pulse"       # 펄스파

class PhaseType(Enum):
    """자극 위상 유형"""
    MONOPHASIC = "monophasic"  # 단상성
    BIPHASIC = "biphasic"      # 양상성
    ASYMMETRIC = "asymmetric"  # 비대칭

class StereoStimulator:
    """
    스테레오 방식의 저주파 자극기를 제어하는 클래스.
    
    특허 10-2459338에 기반하여 구현되었으며,
    두 개 이상의 저주파 자극기에 위상차를 적용하여
    스테레오 효과를 제공합니다.
    """
    
    def __init__(self, num_stimulators=2, 
                 base_frequency=10,
                 waveform_type=WaveformType.SINE,
                 phase_type=PhaseType.BIPHASIC,
                 phase_delay=0.5,
                 amplitude=0.5,
                 pulse_width=0.2):
        """
        StereoStimulator 초기화 함수
        
        매개변수:
            num_stimulators (int): 자극기 수 (기본값: 2)
            base_frequency (float): 기본 주파수 (Hz)
            waveform_type (WaveformType): 파형 유형
            phase_type (PhaseType): 위상 유형
            phase_delay (float): 자극기 간 위상 지연 (초)
            amplitude (float): 신호 진폭 (0-1 사이)
            pulse_width (float): 펄스 폭 (초, 펄스파에만 적용)
        """
        self.num_stimulators = num_stimulators
        self.base_frequency = base_frequency
        self.waveform_type = waveform_type
        self.phase_type = phase_type
        self.phase_delay = phase_delay
        self.amplitude = amplitude
        self.pulse_width = pulse_width
        
        # 자극기별 설정
        self.stimulator_settings = []
        for i in range(num_stimulators):
            self.stimulator_settings.append({
                'id': f"stim_{i+1}",
                'enabled': True,
                'amplitude': amplitude,
                'frequency': base_frequency,
                'delay': i * phase_delay if i > 0 else 0,
                'balance': 1.0  # 자극 강도 밸런스 (0-1)
            })
        
        # 자극 신호 생성 및 제어를 위한 변수
        self.running = False
        self.stimulation_thread = None
        self.music_mode = False
        self.music_features = None
        
        # 생성된 신호 저장
        self.generated_signals = [[] for _ in range(num_stimulators)]
        self.generated_times = []
        
        # 자극 세션 ID
        self.session_id = None
        
        logger.info(f"StereoStimulator 초기화 완료 (자극기 수: {num_stimulators})")
    
    def set_stimulator_balance(self, stim_idx, balance):
        """
        특정 자극기의 강도 밸런스를 설정합니다.
        
        매개변수:
            stim_idx (int): 자극기 인덱스 (0부터 시작)
            balance (float): 강도 밸런스 (0-1 사이)
        """
        if 0 <= stim_idx < self.num_stimulators:
            self.stimulator_settings[stim_idx]['balance'] = max(0, min(1, balance))
            logger.info(f"자극기 {stim_idx}의 밸런스를 {balance}로 설정")
        else:
            logger.warning(f"유효하지 않은 자극기 인덱스: {stim_idx}")
    
    def set_phase_delay(self, delay):
        """
        자극기 간 위상 지연을 설정합니다.
        
        매개변수:
            delay (float): 위상 지연 (초)
        """
        # 위상 지연 범위 검증 (0.1-1.0초, 특허 명세에 따름)
        if 0.1 <= delay <= 1.0:
            self.phase_delay = delay
            
            # 모든 자극기의 지연 업데이트
            for i in range(1, self.num_stimulators):
                self.stimulator_settings[i]['delay'] = i * delay
                
            logger.info(f"위상 지연이 {delay}초로 설정되었습니다.")
        else:
            logger.warning(f"유효하지 않은 위상 지연 값: {delay}, 범위는 0.1-1.0초여야 합니다.")
    
    def set_waveform(self, waveform_type, phase_type=None):
        """
        자극 파형 유형을 설정합니다.
        
        매개변수:
            waveform_type (WaveformType): 파형 유형
            phase_type (PhaseType, optional): 위상 유형
        """
        if isinstance(waveform_type, str):
            try:
                waveform_type = WaveformType(waveform_type)
            except ValueError:
                logger.warning(f"유효하지 않은 파형 유형: {waveform_type}")
                return
        
        self.waveform_type = waveform_type
        
        if phase_type:
            if isinstance(phase_type, str):
                try:
                    phase_type = PhaseType(phase_type)
                except ValueError:
                    logger.warning(f"유효하지 않은 위상 유형: {phase_type}")
                    return
            self.phase_type = phase_type
        
        logger.info(f"파형이 {self.waveform_type.value}로 설정되었으며, 위상 유형은 {self.phase_type.value}입니다.")
    
    def set_frequency(self, frequency):
        """
        기본 주파수를 설정합니다.
        
        매개변수:
            frequency (float): 주파수 (Hz)
        """
        if 1 <= frequency <= 100:  # 일반적인 저주파 자극 범위
            self.base_frequency = frequency
            
            # 모든 자극기의 주파수 업데이트
            for i in range(self.num_stimulators):
                self.stimulator_settings[i]['frequency'] = frequency
                
            logger.info(f"기본 주파수가 {frequency}Hz로 설정되었습니다.")
        else:
            logger.warning(f"유효하지 않은 주파수 값: {frequency}Hz, 범위는 1-100Hz여야 합니다.")
    
    def set_amplitude(self, amplitude, stim_idx=None):
        """
        자극 진폭을 설정합니다.
        
        매개변수:
            amplitude (float): 진폭 (0-1 사이)
            stim_idx (int, optional): 특정 자극기 인덱스. None이면 모든 자극기에 적용
        """
        amplitude = max(0, min(1, amplitude))
        
        if stim_idx is not None:
            if 0 <= stim_idx < self.num_stimulators:
                self.stimulator_settings[stim_idx]['amplitude'] = amplitude
                logger.info(f"자극기 {stim_idx}의 진폭을 {amplitude}로 설정")
            else:
                logger.warning(f"유효하지 않은 자극기 인덱스: {stim_idx}")
        else:
            self.amplitude = amplitude
            for i in range(self.num_stimulators):
                self.stimulator_settings[i]['amplitude'] = amplitude
            logger.info(f"모든 자극기의 진폭이 {amplitude}로 설정되었습니다.")
    
    def enable_stimulator(self, stim_idx, enabled=True):
        """
        특정 자극기를 활성화/비활성화합니다.
        
        매개변수:
            stim_idx (int): 자극기 인덱스
            enabled (bool): 활성화 여부
        """
        if 0 <= stim_idx < self.num_stimulators:
            self.stimulator_settings[stim_idx]['enabled'] = enabled
            status = "활성화" if enabled else "비활성화"
            logger.info(f"자극기 {stim_idx}가 {status}되었습니다.")
        else:
            logger.warning(f"유효하지 않은 자극기 인덱스: {stim_idx}")
    
    def generate_waveform(self, t, settings):
        """
        주어진 시간 t와 설정에 따라 파형을 생성합니다.
        
        매개변수:
            t (float): 시간 (초)
            settings (dict): 자극기 설정
            
        반환값:
            float: 생성된 파형 값 (-1에서 1 사이)
        """
        # 진폭, 주파수, 지연 적용
        amplitude = settings['amplitude'] * settings['balance']
        frequency = settings['frequency']
        delay = settings['delay']
        
        # 지연 적용
        t = t - delay
        
        # 0보다 작은 시간에서는 신호 없음
        if t < 0:
            return 0
        
        # 파형 생성
        if self.waveform_type == WaveformType.SINE:
            # 정현파
            signal = np.sin(2 * np.pi * frequency * t)
        elif self.waveform_type == WaveformType.SQUARE:
            # 구형파
            signal = np.sign(np.sin(2 * np.pi * frequency * t))
        elif self.waveform_type == WaveformType.TRIANGLE:
            # 삼각파
            signal = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        elif self.waveform_type == WaveformType.SAWTOOTH:
            # 톱니파
            signal = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif self.waveform_type == WaveformType.PULSE:
            # 펄스파
            cycle_time = 1.0 / frequency
            t_in_cycle = t % cycle_time
            pulse_duration = self.pulse_width
            signal = 1.0 if t_in_cycle < pulse_duration else 0
        else:
            # 기본값은 정현파
            signal = np.sin(2 * np.pi * frequency * t)
        
        # 위상 유형 적용
        if self.phase_type == PhaseType.MONOPHASIC:
            # 단상성: 양의 값만 유지, 음의 값은 0
            signal = max(0, signal)
        elif self.phase_type == PhaseType.ASYMMETRIC:
            # 비대칭: 음의 값의 진폭을 절반으로
            if signal < 0:
                signal = signal * 0.5
        # BIPHASIC은 기본값으로 그대로 유지
        
        # 진폭 적용
        return signal * amplitude
    
    def start_stimulation(self, duration=300, callback=None):
        """
        자극을 시작합니다.
        
        매개변수:
            duration (int): 자극 지속 시간 (초)
            callback (function): 자극 완료 후 호출될 콜백 함수
        """
        if self.running:
            logger.warning("자극이 이미 실행 중입니다.")
            return
        
        self.running = True
        self.session_id = str(uuid.uuid4())
        
        # 자극 결과 초기화
        self.generated_signals = [[] for _ in range(self.num_stimulators)]
        self.generated_times = []
        
        # 자극 스레드 시작
        self.stimulation_thread = threading.Thread(
            target=self._stimulation_worker,
            args=(duration, callback)
        )
        self.stimulation_thread.daemon = True
        self.stimulation_thread.start()
        
        logger.info(f"자극이 시작되었습니다. 세션 ID: {self.session_id}, 지속 시간: {duration}초")
        return self.session_id
    
    def _stimulation_worker(self, duration, callback):
        """
        자극 생성 및 전송을 처리하는 워커 함수
        
        매개변수:
            duration (int): 자극 지속 시간 (초)
            callback (function): 완료 후 호출될 콜백 함수
        """
        start_time = time.time()
        end_time = start_time + duration
        
        try:
            # 샘플링 속도 (초당 샘플 수)
            sample_rate = 20
            dt = 1.0 / sample_rate
            
            # 자극 생성 및 전송
            current_time = start_time
            while self.running and current_time < end_time:
                # 경과 시간
                elapsed = current_time - start_time
                
                # 모든 자극기에 대한 신호 생성
                signals = []
                for i, settings in enumerate(self.stimulator_settings):
                    if settings['enabled']:
                        # 음악 모드인 경우, 음악 특성 기반 자극 생성
                        if self.music_mode and self.music_features:
                            signal = self._generate_music_based_signal(elapsed, i)
                        else:
                            # 일반 모드, 설정에 따른 자극 생성
                            signal = self.generate_waveform(elapsed, settings)
                    else:
                        signal = 0
                    
                    signals.append(signal)
                    self.generated_signals[i].append(signal)
                
                self.generated_times.append(elapsed)
                
                # 자극 신호 전송 (실제 하드웨어로 전송하는 코드가 여기에 추가될 수 있음)
                self._send_stimulation(signals, elapsed)
                
                # 다음 샘플 시간까지 대기
                time.sleep(dt)
                current_time = time.time()
            
            # 자극 종료 처리
            for i in range(self.num_stimulators):
                # 0 값의 신호 전송으로 자극 중지
                self._send_stimulation([0] * self.num_stimulators, elapsed)
            
            logger.info(f"자극이 완료되었습니다. 세션 ID: {self.session_id}, 총 지속 시간: {time.time() - start_time:.2f}초")
            
            # 자극 결과 저장
            self._save_stimulation_results()
            
            # 콜백 호출
            if callback:
                callback(self.session_id)
                
        except Exception as e:
            logger.error(f"자극 처리 중 오류 발생: {e}")
        finally:
            self.running = False
    
    def _send_stimulation(self, signals, elapsed):
        """
        자극 신호를 하드웨어로 전송합니다.
        
        매개변수:
            signals (list): 각 자극기에 대한 신호 값 리스트
            elapsed (float): 경과 시간
            
        참고: 이 함수는 실제 구현에서 블루투스 또는 기타 통신 방식을 통해
             하드웨어로 신호를 전송하도록 확장될 수 있습니다.
        """
        # 여기서는 로깅만 수행 (실제 구현에서는 하드웨어 통신 코드 추가)
        if elapsed % 5 < 0.1:  # 5초마다 로그 기록
            log_signals = ", ".join([f"{s:.2f}" for s in signals])
            logger.debug(f"자극 신호 전송 (T+{elapsed:.2f}s): [{log_signals}]")
    
    def _generate_music_based_signal(self, elapsed, stim_idx):
        """
        음악 특성에 기반한 자극 신호를 생성합니다.
        
        매개변수:
            elapsed (float): 경과 시간
            stim_idx (int): 자극기 인덱스
            
        반환값:
            float: 생성된 자극 신호
        """
        if not self.music_features or 'beats' not in self.music_features:
            return 0
        
        settings = self.stimulator_settings[stim_idx]
        amplitude = settings['amplitude'] * settings['balance']
        base_freq = settings['frequency']
        
        # 비트 정보에서 현재 시간에 해당하는 비트 강도 찾기
        beats = self.music_features['beats']
        beat_strength = 0
        
        for beat in beats:
            beat_time, intensity = beat
            if abs(elapsed - beat_time) < 0.1:  # 비트 시간 근처 0.1초 내에 있는 경우
                beat_strength = intensity
                break
        
        # 비트 강도에 따른 신호 생성 (비트 강도가 높을수록 진폭 증가)
        if beat_strength > 0:
            # 비트에 따른 주파수 변조
            freq_mod = base_freq * (1 + 0.5 * beat_strength)
            return self.generate_waveform(elapsed, {
                'amplitude': amplitude * (1 + beat_strength),
                'frequency': freq_mod,
                'delay': settings['delay'],
                'balance': 1.0
            })
        else:
            # 일반 신호 생성
            return self.generate_waveform(elapsed, settings)
    
    def stop_stimulation(self):
        """
        자극을 중지합니다.
        """
        if not self.running:
            logger.warning("자극이 실행 중이지 않습니다.")
            return
        
        self.running = False
        
        # 스레드가 종료될 때까지 대기
        if self.stimulation_thread and self.stimulation_thread.is_alive():
            self.stimulation_thread.join(timeout=1.0)
        
        logger.info("자극이 중지되었습니다.")
    
    def set_music_mode(self, enabled=True, music_features=None):
        """
        음악 모드를 설정합니다.
        
        매개변수:
            enabled (bool): 음악 모드 활성화 여부
            music_features (dict): 음악 특성 데이터
        """
        self.music_mode = enabled
        
        if enabled and music_features:
            self.music_features = music_features
            logger.info(f"음악 모드가 활성화되었습니다. {len(music_features.get('beats', []))}개의 비트 정보 로드됨.")
        elif not enabled:
            logger.info("음악 모드가 비활성화되었습니다.")
    
    def extract_music_features(self, audio_data, sample_rate):
        """
        오디오 데이터에서 음악 특성을 추출합니다.
        
        매개변수:
            audio_data (numpy.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        반환값:
            dict: 추출된 음악 특성
        """
        try:
            # 이 함수는 외부 라이브러리(librosa 등)를 사용하여 구현할 수 있습니다.
            # 여기서는 간단한 구현만 제공합니다.
            
            # 진폭 엔벨로프 계산
            frame_size = int(sample_rate * 0.025)  # 25ms 프레임
            hop_size = int(sample_rate * 0.010)    # 10ms 호프
            
            # 프레임별 RMS 에너지
            n_frames = 1 + (len(audio_data) - frame_size) // hop_size
            energy = []
            for i in range(n_frames):
                start = i * hop_size
                end = start + frame_size
                if end > len(audio_data):
                    break
                frame = audio_data[start:end]
                rms = np.sqrt(np.mean(frame**2))
                energy.append(rms)
            
            # 에너지 임계값을 사용하여 비트 검출
            energy = np.array(energy)
            threshold = np.mean(energy) + 0.5 * np.std(energy)
            
            beats = []
            for i in range(1, len(energy) - 1):
                if energy[i] > threshold and energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                    # 비트 시간 및 강도 계산
                    beat_time = i * hop_size / sample_rate
                    intensity = min(1.0, (energy[i] - threshold) / threshold)
                    beats.append((beat_time, intensity))
            
            return {
                'beats': beats,
                'energy': energy.tolist(),
                'times': [i * hop_size / sample_rate for i in range(len(energy))],
                'tempo': len(beats) * 60 / (len(audio_data) / sample_rate) if beats else 0
            }
            
        except Exception as e:
            logger.error(f"음악 특성 추출 중 오류 발생: {e}")
            return {'beats': []}
    
    def _save_stimulation_results(self):
        """
        자극 세션 결과를 저장합니다.
        """
        if not self.generated_times:
            logger.warning("저장할 자극 결과가 없습니다.")
            return
        
        try:
            # 결과 디렉토리 생성
            results_dir = "results/stimulation"
            os.makedirs(results_dir, exist_ok=True)
            
            # 결과 데이터 구성
            result_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'settings': {
                    'num_stimulators': self.num_stimulators,
                    'base_frequency': self.base_frequency,
                    'waveform_type': self.waveform_type.value,
                    'phase_type': self.phase_type.value,
                    'phase_delay': self.phase_delay,
                    'amplitude': self.amplitude,
                    'pulse_width': self.pulse_width,
                    'music_mode': self.music_mode
                },
                'stimulator_settings': self.stimulator_settings,
                'times': self.generated_times
            }
            
            # 신호 데이터 추가
            for i in range(self.num_stimulators):
                result_data[f'signals_{i}'] = self.generated_signals[i]
            
            # JSON 파일로 저장
            filename = f"{results_dir}/session_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(result_data, f)
            
            logger.info(f"자극 결과가 {filename}에 저장되었습니다.")
            
            # 시각화 이미지 생성 및 저장
            self._visualize_stimulation(results_dir)
            
        except Exception as e:
            logger.error(f"자극 결과 저장 중 오류 발생: {e}")
    
    def _visualize_stimulation(self, results_dir):
        """
        자극 신호를 시각화하여 이미지로 저장합니다.
        
        매개변수:
            results_dir (str): 결과 저장 디렉토리
        """
        try:
            # 시각화 생성
            plt.figure(figsize=(12, 8))
            
            # 각 자극기별 신호 플롯
            for i in range(self.num_stimulators):
                plt.subplot(self.num_stimulators, 1, i+1)
                plt.plot(self.generated_times, self.generated_signals[i])
                plt.title(f"자극기 {i+1} (위상 지연: {self.stimulator_settings[i]['delay']:.2f}s)")
                plt.xlabel("시간 (초)")
                plt.ylabel("신호 강도")
                plt.grid(True)
            
            plt.tight_layout()
            
            # 이미지 저장
            filename = f"{results_dir}/session_{self.session_id}_plot.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"자극 시각화가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"자극 시각화 중 오류 발생: {e}")
