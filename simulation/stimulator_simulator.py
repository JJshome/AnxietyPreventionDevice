"""
저주파 자극기 시뮬레이터

특허 10-2459338에 기반한 스테레오 저주파 자극기를 시뮬레이션하는 모듈입니다.
두 개 이상의 자극기에 위상차를 가진 저주파 신호를 생성합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import logging
import uuid
import os
from datetime import datetime
from enum import Enum

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


class StimulatorSimulator:
    """
    저주파 자극기 시뮬레이터
    
    특허 10-2459338에 기반한 스테레오 저주파 자극 시스템을 시뮬레이션합니다.
    """
    
    def __init__(self, num_stimulators=2, visualization=True):
        """
        StimulatorSimulator 초기화
        
        매개변수:
            num_stimulators (int): 자극기 수 (기본값: 2)
            visualization (bool): 시각화 활성화 여부
        """
        self.num_stimulators = num_stimulators
        self.visualization = visualization
        
        # 자극 파라미터
        self.frequency = 10.0  # Hz
        self.waveform_type = WaveformType.SINE
        self.phase_type = PhaseType.BIPHASIC
        self.amplitude = 0.5
        self.pulse_width = 0.1  # 초
        
        # 위상 지연
        self.phase_delay = 0.5  # 초
        
        # 자극기별 설정
        self.stimulator_settings = []
        for i in range(num_stimulators):
            self.stimulator_settings.append({
                'id': f"stim_{i+1}",
                'enabled': True,
                'amplitude': self.amplitude,
                'frequency': self.frequency,
                'delay': i * self.phase_delay if i > 0 else 0,
                'balance': 1.0  # 자극 강도 밸런스 (0-1)
            })
        
        # 시뮬레이션 상태
        self.running = False
        self.simulation_thread = None
        self.listeners = []
        self.session_id = None
        
        # 생성된 신호 저장
        self.generated_signals = [[] for _ in range(num_stimulators)]
        self.times = []
        
        # 시각화용 그래프
        self.fig = None
        self.axes = None
        self.lines = None
        
        logger.info(f"저주파 자극기 시뮬레이터가 초기화되었습니다. (자극기 수: {num_stimulators})")
    
    def add_listener(self, callback):
        """
        자극 신호 리스너를 추가합니다.
        
        매개변수:
            callback (function): 자극 신호가 생성될 때마다 호출될 콜백 함수
        """
        if callback not in self.listeners:
            self.listeners.append(callback)
            logger.info("자극 신호 리스너가 추가되었습니다.")
    
    def remove_listener(self, callback):
        """
        자극 신호 리스너를 제거합니다.
        
        매개변수:
            callback (function): 제거할 콜백 함수
        """
        if callback in self.listeners:
            self.listeners.remove(callback)
            logger.info("자극 신호 리스너가 제거되었습니다.")
    
    def set_frequency(self, frequency):
        """
        자극 주파수를 설정합니다.
        
        매개변수:
            frequency (float): 주파수 (Hz)
        """
        if 1.0 <= frequency <= 100.0:
            self.frequency = frequency
            
            # 모든 자극기의 주파수 업데이트
            for i in range(self.num_stimulators):
                self.stimulator_settings[i]['frequency'] = frequency
                
            logger.info(f"자극 주파수가 {frequency}Hz로 설정되었습니다.")
        else:
            logger.warning(f"유효하지 않은 주파수: {frequency}Hz. 범위는 1-100Hz입니다.")
    
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
        
        logger.info(f"자극 파형이 {self.waveform_type.value}로 설정되었으며, 위상 유형은 {self.phase_type.value}입니다.")
    
    def set_amplitude(self, amplitude, stimulator_idx=None):
        """
        자극 진폭을 설정합니다.
        
        매개변수:
            amplitude (float): 진폭 (0-1)
            stimulator_idx (int, optional): 특정 자극기 인덱스
        """
        amplitude = max(0, min(1, amplitude))
        
        if stimulator_idx is not None:
            if 0 <= stimulator_idx < self.num_stimulators:
                self.stimulator_settings[stimulator_idx]['amplitude'] = amplitude
                logger.info(f"자극기 {stimulator_idx+1}의 진폭이 {amplitude}로 설정되었습니다.")
            else:
                logger.warning(f"유효하지 않은 자극기 인덱스: {stimulator_idx}")
        else:
            # 모든 자극기에 적용
            self.amplitude = amplitude
            for i in range(self.num_stimulators):
                self.stimulator_settings[i]['amplitude'] = amplitude
            logger.info(f"모든 자극기의 진폭이 {amplitude}로 설정되었습니다.")
    
    def set_phase_delay(self, delay):
        """
        자극기 간 위상 지연을 설정합니다.
        
        매개변수:
            delay (float): 위상 지연 (초)
        """
        if 0.1 <= delay <= 1.0:
            self.phase_delay = delay
            
            # 각 자극기의 지연 업데이트
            for i in range(self.num_stimulators):
                if i > 0:  # 첫 번째 자극기는 지연 없음
                    self.stimulator_settings[i]['delay'] = i * delay
            
            logger.info(f"위상 지연이 {delay}초로 설정되었습니다.")
        else:
            logger.warning(f"유효하지 않은 위상 지연: {delay}초. 범위는 0.1-1.0초입니다.")
    
    def set_balance(self, balances):
        """
        자극기별 강도 밸런스를 설정합니다.
        
        매개변수:
            balances (list): 자극기별 밸런스 값 리스트
        """
        if isinstance(balances, dict):
            for idx, balance in balances.items():
                idx = int(idx) if isinstance(idx, str) else idx
                if 0 <= idx < self.num_stimulators:
                    self.stimulator_settings[idx]['balance'] = max(0, min(1, balance))
        elif isinstance(balances, list):
            for i, balance in enumerate(balances):
                if i < self.num_stimulators:
                    self.stimulator_settings[i]['balance'] = max(0, min(1, balance))
        
        logger.info("자극기 밸런스가 설정되었습니다.")
    
    def enable_stimulator(self, stimulator_idx, enabled=True):
        """
        특정 자극기를 활성화/비활성화합니다.
        
        매개변수:
            stimulator_idx (int): 자극기 인덱스
            enabled (bool): 활성화 여부
        """
        if 0 <= stimulator_idx < self.num_stimulators:
            self.stimulator_settings[stimulator_idx]['enabled'] = enabled
            status = "활성화" if enabled else "비활성화"
            logger.info(f"자극기 {stimulator_idx+1}가 {status}되었습니다.")
        else:
            logger.warning(f"유효하지 않은 자극기 인덱스: {stimulator_idx}")
    
    def generate_signal(self, t, settings):
        """
        주어진 시간 t에 대한 자극 신호를 생성합니다.
        
        매개변수:
            t (float): 시간 (초)
            settings (dict): 자극기 설정
            
        반환값:
            float: 생성된 신호 값
        """
        if not settings['enabled']:
            return 0.0
        
        # 설정 적용
        amplitude = settings['amplitude'] * settings['balance']
        frequency = settings['frequency']
        delay = settings['delay']
        
        # 지연 적용
        t_adjusted = t - delay
        
        # 0보다 작은 시간에서는 신호 없음
        if t_adjusted < 0:
            return 0.0
        
        # 파형 생성
        if self.waveform_type == WaveformType.SINE:
            # 정현파
            signal = np.sin(2 * np.pi * frequency * t_adjusted)
        elif self.waveform_type == WaveformType.SQUARE:
            # 구형파
            signal = np.sign(np.sin(2 * np.pi * frequency * t_adjusted))
        elif self.waveform_type == WaveformType.TRIANGLE:
            # 삼각파
            signal = 2 * np.abs(2 * (frequency * t_adjusted - np.floor(frequency * t_adjusted + 0.5))) - 1
        elif self.waveform_type == WaveformType.SAWTOOTH:
            # 톱니파
            signal = 2 * (frequency * t_adjusted - np.floor(frequency * t_adjusted)) - 1
        elif self.waveform_type == WaveformType.PULSE:
            # 펄스파
            cycle_time = 1.0 / frequency
            t_in_cycle = t_adjusted % cycle_time
            signal = 1.0 if t_in_cycle < self.pulse_width else 0.0
        else:
            # 기본값은 정현파
            signal = np.sin(2 * np.pi * frequency * t_adjusted)
        
        # 위상 유형 적용
        if self.phase_type == PhaseType.MONOPHASIC:
            # 단상성: 음의 값은 0으로 설정
            signal = max(0, signal)
        elif self.phase_type == PhaseType.ASYMMETRIC:
            # 비대칭: 음의 값의 진폭을 절반으로
            if signal < 0:
                signal = signal * 0.5
        # BIPHASIC은 그대로 사용
        
        # 진폭 적용
        return signal * amplitude
    
    def start_stimulation(self, duration=300, callback=None):
        """
        자극을 시작합니다.
        
        매개변수:
            duration (int): 자극 지속 시간 (초)
            callback (function): 자극 완료 후 호출될 콜백 함수
        
        반환값:
            str: 자극 세션 ID
        """
        if self.running:
            logger.warning("자극이 이미 실행 중입니다.")
            return None
        
        self.running = True
        self.session_id = str(uuid.uuid4())
        
        # 결과 초기화
        self.generated_signals = [[] for _ in range(self.num_stimulators)]
        self.times = []
        
        # 시각화 준비
        if self.visualization:
            self._setup_visualization()
        
        # 자극 스레드 시작
        self.simulation_thread = threading.Thread(
            target=self._simulation_worker,
            args=(duration, callback)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        logger.info(f"자극이 시작되었습니다. 세션 ID: {self.session_id}, 지속 시간: {duration}초")
        return self.session_id
    
    def _simulation_worker(self, duration, callback):
        """
        자극 시뮬레이션 워커 함수
        
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
            
            # 시뮬레이션 루프
            current_time = start_time
            while self.running and current_time < end_time:
                # 시뮬레이션 시간 (0부터 시작)
                elapsed = current_time - start_time
                
                # 각 자극기의 신호 생성
                signals = []
                for i, settings in enumerate(self.stimulator_settings):
                    signal = self.generate_signal(elapsed, settings)
                    signals.append(signal)
                    
                    if self.running:  # 중지된 경우 기록하지 않음
                        self.generated_signals[i].append(signal)
                
                if self.running:
                    self.times.append(elapsed)
                
                # 리스너에게 알림
                for listener in self.listeners:
                    try:
                        listener({
                            "signals": signals,
                            "time": elapsed,
                            "session_id": self.session_id,
                            "settings": self.stimulator_settings
                        })
                    except Exception as e:
                        logger.error(f"리스너 호출 중 오류 발생: {e}")
                
                # 시각화 업데이트
                if self.visualization and self.fig is not None and plt.fignum_exists(self.fig.number):
                    self._update_visualization()
                
                # 다음 샘플까지 대기
                time.sleep(dt)
                current_time = time.time()
            
            # 종료 처리
            if current_time >= end_time:
                logger.info(f"자극이 정상적으로 완료되었습니다. 세션 ID: {self.session_id}")
            
            # 콜백 호출
            if callback:
                callback({
                    "session_id": self.session_id,
                    "duration": duration,
                    "actual_duration": current_time - start_time
                })
            
            # 결과 저장
            if self.running:  # 사용자에 의한 중지가 아닌 경우
                self._save_results()
            
        except Exception as e:
            logger.error(f"자극 시뮬레이션 중 오류 발생: {e}")
        finally:
            self.running = False
            
            # 시각화 창 닫기
            if self.visualization and self.fig is not None:
                try:
                    plt.close(self.fig)
                except:
                    pass
                self.fig = None
    
    def stop_stimulation(self):
        """
        자극을 중지합니다.
        
        반환값:
            bool: 성공 여부
        """
        if not self.running:
            logger.warning("자극이 실행 중이지 않습니다.")
            return False
        
        self.running = False
        
        # 스레드 종료 대기
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        
        logger.info("자극이 중지되었습니다.")
        return True
    
    def _setup_visualization(self):
        """
        자극 시각화를 위한 그래프를 설정합니다.
        """
        try:
            # 이전 그래프 닫기
            if self.fig is not None:
                plt.close(self.fig)
            
            # 새 그래프 생성
            self.fig, self.axes = plt.subplots(self.num_stimulators, 1, figsize=(10, 6))
            
            # 단일 자극기인 경우 축을 리스트로 변환
            if self.num_stimulators == 1:
                self.axes = [self.axes]
            
            # 각 자극기별 선 객체 생성
            self.lines = []
            for i in range(self.num_stimulators):
                line, = self.axes[i].plot([], [], 'b-')
                self.lines.append(line)
                
                self.axes[i].set_xlim(0, 2)  # 초기 시간 범위 (2초)
                self.axes[i].set_ylim(-1.1, 1.1)  # 신호 범위
                self.axes[i].set_title(f"자극기 {i+1}")
                self.axes[i].set_xlabel("시간 (초)")
                self.axes[i].set_ylabel("신호")
                self.axes[i].grid(True)
            
            plt.tight_layout()
            plt.ion()  # 대화형 모드 활성화
            plt.show(block=False)
            
        except Exception as e:
            logger.error(f"시각화 설정 중 오류 발생: {e}")
            self.visualization = False
    
    def _update_visualization(self):
        """
        자극 시각화를 업데이트합니다.
        """
        try:
            # 데이터가 없으면 업데이트하지 않음
            if not self.times:
                return
            
            # 표시할 시간 범위 (최근 3초)
            current_time = self.times[-1]
            xmin = max(0, current_time - 3)
            xmax = current_time + 0.5
            
            # 표시할 데이터 인덱스 찾기
            start_idx = 0
            for i, t in enumerate(self.times):
                if t >= xmin:
                    start_idx = i
                    break
            
            # 각 자극기별 그래프 업데이트
            for i in range(self.num_stimulators):
                self.lines[i].set_data(
                    self.times[start_idx:],
                    self.generated_signals[i][start_idx:]
                )
                self.axes[i].set_xlim(xmin, xmax)
                
                # 자극기 상태에 따라 제목 업데이트
                status = "활성" if self.stimulator_settings[i]['enabled'] else "비활성"
                amp = self.stimulator_settings[i]['amplitude']
                balance = self.stimulator_settings[i]['balance']
                self.axes[i].set_title(
                    f"자극기 {i+1} ({status}, 진폭: {amp:.2f}, 밸런스: {balance:.2f})"
                )
            
            # 그래프 업데이트
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"시각화 업데이트 중 오류 발생: {e}")
    
    def _save_results(self):
        """
        자극 결과를 저장합니다.
        """
        try:
            # 저장 디렉토리 생성
            results_dir = "results/stimulation"
            os.makedirs(results_dir, exist_ok=True)
            
            # 결과 구성
            results = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "num_stimulators": self.num_stimulators,
                    "frequency": self.frequency,
                    "waveform_type": self.waveform_type.value,
                    "phase_type": self.phase_type.value,
                    "amplitude": self.amplitude,
                    "phase_delay": self.phase_delay,
                    "pulse_width": self.pulse_width
                },
                "stimulator_settings": [
                    {k: v for k, v in s.items()}
                    for s in self.stimulator_settings
                ],
                "times": self.times
            }
            
            # 신호 데이터 추가
            for i in range(self.num_stimulators):
                results[f"signals_{i}"] = self.generated_signals[i]
            
            # JSON 파일로 저장
            filename = f"{results_dir}/session_{self.session_id}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(results, f, default=str)
            
            logger.info(f"자극 결과가 {filename}에 저장되었습니다.")
            
            # 시각화 이미지 저장
            self._save_visualization(results_dir)
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
    
    def _save_visualization(self, results_dir):
        """
        자극 시각화를 이미지로 저장합니다.
        
        매개변수:
            results_dir (str): 결과 저장 디렉토리
        """
        try:
            if not self.times or not self.generated_signals[0]:
                return
            
            # 새 그래프 생성
            fig, axes = plt.subplots(self.num_stimulators, 1, figsize=(10, 6))
            
            # 단일 자극기인 경우 축을 리스트로 변환
            if self.num_stimulators == 1:
                axes = [axes]
            
            # 각 자극기별 그래프 그리기
            for i in range(self.num_stimulators):
                axes[i].plot(self.times, self.generated_signals[i])
                axes[i].set_title(f"자극기 {i+1} (위상 지연: {self.stimulator_settings[i]['delay']:.2f}초)")
                axes[i].set_xlabel("시간 (초)")
                axes[i].set_ylabel("신호")
                axes[i].grid(True)
            
            plt.tight_layout()
            
            # 이미지 저장
            filename = f"{results_dir}/session_{self.session_id}_plot.png"
            plt.savefig(filename)
            plt.close(fig)
            
            logger.info(f"자극 시각화가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"시각화 저장 중 오류 발생: {e}")
    
    def visualize(self, duration=None):
        """
        생성된 자극 신호를 시각화합니다.
        
        매개변수:
            duration (float): 시각화할 시간 (초)
        """
        if not self.times:
            logger.warning("시각화할 데이터가 없습니다.")
            return
        
        try:
            # 표시할 데이터 범위 결정
            if duration is None:
                start_idx = 0
                times = self.times
                signals = [signal for signal in self.generated_signals]
            else:
                # 최근 duration 초 데이터만 표시
                current_time = self.times[-1]
                start_time = current_time - duration
                
                start_idx = 0
                for i, t in enumerate(self.times):
                    if t >= start_time:
                        start_idx = i
                        break
                
                times = self.times[start_idx:]
                signals = [signal[start_idx:] for signal in self.generated_signals]
            
            # 그래프 생성
            fig, axes = plt.subplots(self.num_stimulators, 1, figsize=(10, 6))
            
            # 단일 자극기인 경우 축을 리스트로 변환
            if self.num_stimulators == 1:
                axes = [axes]
            
            # 각 자극기별 그래프 그리기
            for i in range(self.num_stimulators):
                axes[i].plot(times, signals[i])
                axes[i].set_title(f"자극기 {i+1} (위상 지연: {self.stimulator_settings[i]['delay']:.2f}초)")
                axes[i].set_xlabel("시간 (초)")
                axes[i].set_ylabel("신호")
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"시각화 중 오류 발생: {e}")


# 테스트 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    simulator = StimulatorSimulator(num_stimulators=2)
    
    def stimulation_callback(data):
        logger.info(f"자극 완료: {data['session_id']}")
    
    # 자극 파라미터 설정
    simulator.set_frequency(15.0)
    simulator.set_waveform(WaveformType.SINE, PhaseType.BIPHASIC)
    simulator.set_phase_delay(0.5)
    
    # 밸런스 설정 (첫번째 자극기는 100%, 두번째는 50%)
    simulator.set_balance([1.0, 0.5])
    
    # 10초간 자극 시작
    simulator.start_stimulation(duration=10, callback=stimulation_callback)
    
    # 메인 스레드 대기
    try:
        while simulator.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        simulator.stop_stimulation()
    
    # 결과 시각화
    simulator.visualize()
