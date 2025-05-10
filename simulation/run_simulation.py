"""
불안장애 예방 시스템 시뮬레이션 실행기

ECG 센서와 저주파 자극기를 시뮬레이션하여 불안장애 예방 시스템을 테스트하는 모듈입니다.
"""

import sys
import os
import time
import threading
import logging
import argparse
import numpy as np
import matplotlib
import json
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime

# 상위 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 시뮬레이터 모듈 임포트
from simulation.ecg_simulator import ECGSimulator, ECGPatternType
from simulation.stimulator_simulator import StimulatorSimulator, WaveformType, PhaseType

# 시스템 모듈 임포트
from src.anxiety_prevention_controller import AnxietyPreventionController, SystemState, OperationMode
from src.ecg_sensor.ecg_interface import ECGInterface
from src.ecg_sensor.data_processor import ECGDataProcessor
from src.hrv_analyzer.hrv_anxiety_predictor import HRVAnxietyPredictor
from src.stimulation_controller.stereo_stimulator import StereoStimulator

logger = logging.getLogger(__name__)


class SimulatedECGInterface(ECGInterface):
    """
    ECG 시뮬레이터를 사용하는 ECG 인터페이스 구현
    
    실제 ECG 센서 대신 시뮬레이터를 사용하여 ECGInterface를 구현합니다.
    """
    
    def __init__(self, ecg_simulator, sampling_rate=256, buffer_size=None, bluetooth_enabled=False):
        """
        SimulatedECGInterface 초기화
        
        매개변수:
            ecg_simulator (ECGSimulator): ECG 시뮬레이터 인스턴스
            sampling_rate (int): 샘플링 레이트
            buffer_size (int): 버퍼 크기
            bluetooth_enabled (bool): 블루투스 활성화 여부 (시뮬레이션에서는 무시됨)
        """
        super().__init__(sampling_rate, buffer_size, bluetooth_enabled)
        self.simulator = ecg_simulator
        self.data_buffer = []
        self.connected = False
        
        # 데이터 리스너 등록
        self.simulator.add_listener(self._on_ecg_data)
    
    def _on_ecg_data(self, data):
        """
        ECG 시뮬레이터에서 데이터 수신 시 호출되는 콜백
        
        매개변수:
            data (dict): ECG 데이터
        """
        if self.connected:
            # 데이터 버퍼에 추가
            self.data_buffer.extend(data["data"])
            
            # 버퍼 사이즈 제한
            if self.buffer_size and len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
    
    def connect(self):
        """
        ECG 센서(시뮬레이터) 연결
        
        반환값:
            bool: 성공 여부
        """
        self.connected = True
        self.simulator.start()
        logger.info("시뮬레이션된 ECG 센서가 연결되었습니다.")
        return True
    
    def disconnect(self):
        """
        ECG 센서(시뮬레이터) 연결 해제
        
        반환값:
            bool: 성공 여부
        """
        self.connected = False
        self.simulator.stop()
        logger.info("시뮬레이션된 ECG 센서 연결이 해제되었습니다.")
        return True
    
    def is_connected(self):
        """
        ECG 센서(시뮬레이터) 연결 상태 확인
        
        반환값:
            bool: 연결 여부
        """
        return self.connected
    
    def read_data(self):
        """
        버퍼에서 데이터 읽기
        
        반환값:
            list: ECG 데이터
        """
        if not self.connected or not self.data_buffer:
            return []
        
        # 현재 버퍼 복사 후 비움
        data = self.data_buffer.copy()
        self.data_buffer = []
        
        return data


class SimulatedStereoStimulator(StereoStimulator):
    """
    자극기 시뮬레이터를 사용하는 StereoStimulator 구현
    
    실제 저주파 자극기 대신 시뮬레이터를 사용하여 StereoStimulator를 구현합니다.
    """
    
    def __init__(self, stimulator_simulator, num_stimulators=2, base_frequency=10, 
                 waveform_type=WaveformType.SINE, phase_type=PhaseType.BIPHASIC, 
                 phase_delay=0.5, amplitude=0.5):
        """
        SimulatedStereoStimulator 초기화
        
        매개변수:
            stimulator_simulator (StimulatorSimulator): 자극기 시뮬레이터 인스턴스
            num_stimulators (int): 자극기 수
            base_frequency (float): 기본 주파수
            waveform_type (WaveformType): 파형 유형
            phase_type (PhaseType): 위상 유형
            phase_delay (float): 위상 지연
            amplitude (float): 진폭
        """
        super().__init__(num_stimulators, base_frequency, waveform_type, phase_type, phase_delay, amplitude)
        self.simulator = stimulator_simulator
        
        # 시뮬레이터에 설정 적용
        self.simulator.set_frequency(base_frequency)
        self.simulator.set_waveform(waveform_type.value, phase_type.value)
        self.simulator.set_phase_delay(phase_delay)
        self.simulator.set_amplitude(amplitude)
    
    def start_stimulation(self, params=None):
        """
        자극 시작
        
        매개변수:
            params (dict): 자극 매개변수
            
        반환값:
            bool: 성공 여부
        """
        # 매개변수 적용
        if params:
            if 'frequency' in params:
                self.simulator.set_frequency(params['frequency'])
            if 'waveform_type' in params:
                self.simulator.set_waveform(params['waveform_type'])
            if 'phase_type' in params:
                self.simulator.set_waveform(self.simulator.waveform_type, params['phase_type'])
            if 'amplitude' in params:
                self.simulator.set_amplitude(params['amplitude'])
            if 'phase_delay' in params:
                self.simulator.set_phase_delay(params['phase_delay'])
            if 'duration' in params:
                duration = params['duration']
            else:
                duration = 180  # 기본 지속 시간 3분
        else:
            duration = 180
        
        # 자극 시작
        session_id = self.simulator.start_stimulation(duration=duration)
        return bool(session_id)
    
    def stop_stimulation(self):
        """
        자극 중지
        
        반환값:
            bool: 성공 여부
        """
        return self.simulator.stop_stimulation()
    
    def set_phase_delay(self, delay):
        """
        위상 지연 설정
        
        매개변수:
            delay (float): 위상 지연
            
        반환값:
            bool: 성공 여부
        """
        self.phase_delay = delay
        self.simulator.set_phase_delay(delay)
        return True
    
    def set_balance(self, balance_params):
        """
        밸런스 설정
        
        매개변수:
            balance_params (dict): 밸런스 매개변수
            
        반환값:
            bool: 성공 여부
        """
        self.simulator.set_balance(balance_params)
        return True
    
    def register_stimulator(self, stimulator_id, stimulator_interface):
        """
        자극기 등록 (시뮬레이션에서는 사용하지 않음)
        
        매개변수:
            stimulator_id (str): 자극기 ID
            stimulator_interface: 자극기 인터페이스
            
        반환값:
            bool: 성공 여부
        """
        logger.info(f"시뮬레이션 모드에서는 자극기 등록이 무시됩니다: {stimulator_id}")
        return True
    
    def get_stimulators(self):
        """
        등록된 자극기 목록 반환
        
        반환값:
            list: 자극기 ID 목록
        """
        return [f"stim_{i+1}" for i in range(self.num_stimulators)]
    
    def get_stimulator_status(self, stimulator_id):
        """
        자극기 상태 반환
        
        매개변수:
            stimulator_id (str): 자극기 ID
            
        반환값:
            dict: 자극기 상태
        """
        for i, stim_id in enumerate(self.get_stimulators()):
            if stim_id == stimulator_id:
                return {
                    "id": stimulator_id,
                    "connected": True,
                    "battery": 80,
                    "enabled": self.simulator.stimulator_settings[i]['enabled'],
                    "balance": self.simulator.stimulator_settings[i]['balance'],
                    "amplitude": self.simulator.stimulator_settings[i]['amplitude'],
                    "delay": self.simulator.stimulator_settings[i]['delay']
                }
        return None


class SimulationApp:
    """
    불안장애 예방 시스템 시뮬레이션 애플리케이션
    
    ECG 시뮬레이터, 자극기 시뮬레이터, 불안장애 예방 컨트롤러를 통합하여
    전체 시스템을 시뮬레이션하는 GUI 애플리케이션을 제공합니다.
    """
    
    def __init__(self, root=None, headless=False):
        """
        SimulationApp 초기화
        
        매개변수:
            root (tk.Tk): Tkinter 루트 윈도우 (헤드리스 모드에서는 None)
            headless (bool): 헤드리스 모드 여부
        """
        self.headless = headless
        self.root = root
        
        # 시뮬레이터 생성
        self.ecg_simulator = ECGSimulator(sampling_rate=256, noise_level=0.05)
        self.stimulator_simulator = StimulatorSimulator(
            num_stimulators=2, 
            visualization=not headless
        )
        
        # 시뮬레이션된 인터페이스 생성
        self.ecg_interface = SimulatedECGInterface(
            self.ecg_simulator, 
            sampling_rate=256, 
            buffer_size=256 * 60 * 5  # 5분 데이터
        )
        self.stimulator_interface = SimulatedStereoStimulator(
            self.stimulator_simulator, 
            num_stimulators=2
        )
        
        # 컨트롤러 생성 및 인터페이스 등록
        self.controller = AnxietyPreventionController()
        self.controller._ecg_sensor = self.ecg_interface
        self.controller._stimulator = self.stimulator_interface
        
        # 시뮬레이션 상태
        self.running = False
        self.current_scenario = None
        self.scenario_thread = None
        
        # GUI 초기화 (헤드리스 모드가 아닌 경우)
        if not headless:
            self._init_gui()
    
    def _init_gui(self):
        """
        GUI 초기화
        """
        self.root.title("불안장애 예방 시스템 시뮬레이션")
        self.root.geometry("1000x800")
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 상단 제어 프레임
        control_frame = ttk.LabelFrame(main_frame, text="시뮬레이션 제어", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # 시나리오 선택
        scenario_frame = ttk.Frame(control_frame)
        scenario_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scenario_frame, text="시나리오:").pack(side=tk.LEFT, padx=5)
        
        self.scenario_var = tk.StringVar(value="normal")
        scenarios = [
            ("정상 상태", "normal"),
            ("불안 수준 증가", "increasing_anxiety"),
            ("갑작스런 불안 발작", "sudden_anxiety"),
            ("불안 발작 후 회복", "recovery_after_anxiety")
        ]
        
        for text, value in scenarios:
            ttk.Radiobutton(
                scenario_frame, 
                text=text, 
                value=value, 
                variable=self.scenario_var
            ).pack(side=tk.LEFT, padx=5)
        
        # 시작/중지 버튼
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="시작", 
            command=self.start_simulation
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="중지", 
            command=self.stop_simulation, 
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.manual_stim_button = ttk.Button(
            button_frame, 
            text="수동 자극 실행", 
            command=self.trigger_manual_stimulation, 
            state=tk.DISABLED
        )
        self.manual_stim_button.pack(side=tk.LEFT, padx=5)
        
        # 모드 선택
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="작동 모드:").pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="automatic")
        modes = [
            ("자동", "automatic"),
            ("수동", "manual"),
            ("예방", "prevention"),
            ("훈련", "training")
        ]
        
        for text, value in modes:
            ttk.Radiobutton(
                mode_frame, 
                text=text, 
                value=value, 
                variable=self.mode_var, 
                command=self.change_mode
            ).pack(side=tk.LEFT, padx=5)
        
        # 심전도 설정 프레임
        ecg_frame = ttk.LabelFrame(main_frame, text="ECG 시뮬레이터 설정", padding=10)
        ecg_frame.pack(fill=tk.X, pady=5)
        
        # 심박수 설정
        hr_frame = ttk.Frame(ecg_frame)
        hr_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hr_frame, text="심박수 (bpm):").pack(side=tk.LEFT, padx=5)
        
        self.hr_var = tk.IntVar(value=70)
        hr_scale = ttk.Scale(
            hr_frame, 
            from_=40, 
            to=200, 
            orient=tk.HORIZONTAL, 
            variable=self.hr_var, 
            length=200
        )
        hr_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        hr_label = ttk.Label(hr_frame, textvariable=self.hr_var, width=4)
        hr_label.pack(side=tk.LEFT, padx=5)
        
        # HRV 수준 설정
        hrv_frame = ttk.Frame(ecg_frame)
        hrv_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hrv_frame, text="HRV 수준:").pack(side=tk.LEFT, padx=5)
        
        self.hrv_var = tk.DoubleVar(value=0.1)
        hrv_scale = ttk.Scale(
            hrv_frame, 
            from_=0.01, 
            to=0.3, 
            orient=tk.HORIZONTAL, 
            variable=self.hrv_var, 
            length=200
        )
        hrv_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.hrv_label = ttk.Label(hrv_frame, text="0.10")
        self.hrv_label.pack(side=tk.LEFT, padx=5)
        
        # HRV 변경 시 레이블 업데이트
        def update_hrv_label(*args):
            self.hrv_label.config(text=f"{self.hrv_var.get():.2f}")
        
        self.hrv_var.trace_add("write", update_hrv_label)
        
        # ECG 패턴 설정
        pattern_frame = ttk.Frame(ecg_frame)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pattern_frame, text="ECG 패턴:").pack(side=tk.LEFT, padx=5)
        
        self.pattern_var = tk.StringVar(value="normal")
        patterns = [
            ("정상", "normal"),
            ("높은 심박수", "elevated_hr"),
            ("낮은 HRV", "reduced_hrv"),
            ("불안", "anxiety"),
            ("부정맥", "arrhythmia")
        ]
        
        for text, value in patterns:
            ttk.Radiobutton(
                pattern_frame, 
                text=text, 
                value=value, 
                variable=self.pattern_var
            ).pack(side=tk.LEFT, padx=5)
        
        # 자극기 설정 프레임
        stim_frame = ttk.LabelFrame(main_frame, text="자극기 설정", padding=10)
        stim_frame.pack(fill=tk.X, pady=5)
        
        # 자극 주파수 설정
        freq_frame = ttk.Frame(stim_frame)
        freq_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(freq_frame, text="주파수 (Hz):").pack(side=tk.LEFT, padx=5)
        
        self.freq_var = tk.DoubleVar(value=10.0)
        freq_scale = ttk.Scale(
            freq_frame, 
            from_=1.0, 
            to=50.0, 
            orient=tk.HORIZONTAL, 
            variable=self.freq_var, 
            length=200
        )
        freq_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.freq_label = ttk.Label(freq_frame, text="10.0")
        self.freq_label.pack(side=tk.LEFT, padx=5)
        
        # 주파수 변경 시 레이블 업데이트
        def update_freq_label(*args):
            self.freq_label.config(text=f"{self.freq_var.get():.1f}")
        
        self.freq_var.trace_add("write", update_freq_label)
        
        # 위상 지연 설정
        delay_frame = ttk.Frame(stim_frame)
        delay_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(delay_frame, text="위상 지연 (초):").pack(side=tk.LEFT, padx=5)
        
        self.delay_var = tk.DoubleVar(value=0.5)
        delay_scale = ttk.Scale(
            delay_frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.delay_var, 
            length=200
        )
        delay_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.delay_label = ttk.Label(delay_frame, text="0.5")
        self.delay_label.pack(side=tk.LEFT, padx=5)
        
        # 위상 지연 변경 시 레이블 업데이트
        def update_delay_label(*args):
            self.delay_label.config(text=f"{self.delay_var.get():.1f}")
        
        self.delay_var.trace_add("write", update_delay_label)
        
        # 파형 유형 설정
        waveform_frame = ttk.Frame(stim_frame)
        waveform_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(waveform_frame, text="파형 유형:").pack(side=tk.LEFT, padx=5)
        
        self.waveform_var = tk.StringVar(value="sine")
        waveforms = [
            ("정현파", "sine"),
            ("구형파", "square"),
            ("삼각파", "triangle"),
            ("톱니파", "sawtooth"),
            ("펄스파", "pulse")
        ]
        
        for text, value in waveforms:
            ttk.Radiobutton(
                waveform_frame, 
                text=text, 
                value=value, 
                variable=self.waveform_var
            ).pack(side=tk.LEFT, padx=5)
        
        # 설정 적용 버튼
        apply_frame = ttk.Frame(stim_frame)
        apply_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            apply_frame, 
            text="ECG 설정 적용", 
            command=self.apply_ecg_settings
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            apply_frame, 
            text="자극기 설정 적용", 
            command=self.apply_stimulator_settings
        ).pack(side=tk.LEFT, padx=5)
        
        # 시스템 상태 프레임
        status_frame = ttk.LabelFrame(main_frame, text="시스템 상태", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=80, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)
        
        # 그래프 프레임
        graph_frame = ttk.LabelFrame(main_frame, text="모니터링", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Matplotlib 그래프 추가
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 그래프 초기화
        self.ax_ecg = self.fig.add_subplot(3, 1, 1)
        self.ax_ecg.set_title("ECG 신호")
        self.ax_ecg.set_ylabel("진폭")
        self.ax_ecg.grid(True)
        
        self.ax_hrv = self.fig.add_subplot(3, 1, 2)
        self.ax_hrv.set_title("HRV 지표 및 불안 점수")
        self.ax_hrv.set_ylabel("값")
        self.ax_hrv.grid(True)
        
        self.ax_stim = self.fig.add_subplot(3, 1, 3)
        self.ax_stim.set_title("자극 신호")
        self.ax_stim.set_xlabel("시간 (초)")
        self.ax_stim.set_ylabel("진폭")
        self.ax_stim.grid(True)
        
        self.fig.tight_layout()
        
        # 그래프 업데이트 타이머
        self.update_plot()
        
        # 상태 업데이트 타이머
        self.update_status()
    
    def apply_ecg_settings(self):
        """
        ECG 설정을 시뮬레이터에 적용
        """
        if not self.ecg_simulator:
            return
        
        hr = self.hr_var.get()
        hrv_level = self.hrv_var.get()
        pattern = self.pattern_var.get()
        
        self.ecg_simulator.set_pattern(pattern, heart_rate=hr, hrv_level=hrv_level)
        
        self.log_status(f"ECG 설정이 적용되었습니다. HR={hr}bpm, HRV={hrv_level:.2f}, 패턴={pattern}")
    
    def apply_stimulator_settings(self):
        """
        자극기 설정을 시뮬레이터에 적용
        """
        if not self.stimulator_simulator:
            return
        
        frequency = self.freq_var.get()
        delay = self.delay_var.get()
        waveform = self.waveform_var.get()
        
        self.stimulator_simulator.set_frequency(frequency)
        self.stimulator_simulator.set_phase_delay(delay)
        self.stimulator_simulator.set_waveform(waveform)
        
        self.log_status(f"자극기 설정이 적용되었습니다. 주파수={frequency}Hz, 위상 지연={delay}초, 파형={waveform}")
    
    def log_status(self, message):
        """
        상태 텍스트에 메시지 추가
        
        매개변수:
            message (str): 로그 메시지
        """
        if self.headless:
            logger.info(message)
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, log_message)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def change_mode(self):
        """
        작동 모드 변경
        """
        mode = self.mode_var.get()
        self.controller.set_operation_mode(mode)
        self.log_status(f"작동 모드가 변경되었습니다: {mode}")
    
    def update_status(self):
        """
        상태 정보 업데이트
        """
        if self.running and not self.headless:
            # 컨트롤러 상태 가져오기
            status = self.controller.get_status()
            
            # 상태 텍스트 업데이트 (너무 자주 하지 않도록)
            if status and hasattr(self, 'last_status_update') and time.time() - self.last_status_update >= 1.0:
                state = status.get('state', 'unknown')
                
                # 불안 점수 표시
                anxiety_score = status.get('anxiety_score')
                if anxiety_score is not None:
                    anxiety_status = f", 불안 점수: {anxiety_score:.2f}"
                else:
                    anxiety_status = ""
                
                self.log_status(f"시스템 상태: {state}{anxiety_status}")
                self.last_status_update = time.time()
            
            # 1초 후 다시 호출
            self.root.after(1000, self.update_status)
    
    def update_plot(self):
        """
        그래프 업데이트
        """
        if self.running and not self.headless:
            try:
                # ECG 데이터 업데이트
                try:
                    ecg_data, timestamps = self.ecg_simulator.get_data(duration=5)
                    if len(ecg_data) > 0 and len(timestamps) > 0:
                        self.ax_ecg.clear()
                        self.ax_ecg.plot(timestamps - timestamps[0], ecg_data)
                        self.ax_ecg.set_title("ECG 신호")
                        self.ax_ecg.set_ylabel("진폭")
                        self.ax_ecg.grid(True)
                except Exception as e:
                    logger.error(f"ECG 그래프 업데이트 중 오류: {e}")
                
                # HRV 및 불안 점수 업데이트
                try:
                    if hasattr(self.controller, 'anxiety_scores') and len(self.controller.anxiety_scores) > 0:
                        self.ax_hrv.clear()
                        
                        # 타임스탬프 변환 (0부터 시작)
                        if self.controller.timestamps:
                            times = [t - self.controller.timestamps[0] for t in self.controller.timestamps]
                        else:
                            times = list(range(len(self.controller.anxiety_scores)))
                        
                        # 불안 점수 플롯
                        self.ax_hrv.plot(times, self.controller.anxiety_scores, 'r-', label='불안 점수')
                        
                        # 자극 시점 표시
                        if self.controller.last_stimulation_time > 0:
                            stim_time = self.controller.last_stimulation_time - self.controller.timestamps[0] if self.controller.timestamps else 0
                            self.ax_hrv.axvline(x=stim_time, color='g', linestyle='--', label='자극 시점')
                        
                        self.ax_hrv.set_title("불안 점수")
                        self.ax_hrv.set_ylabel("점수")
                        self.ax_hrv.set_ylim(0, 1)
                        self.ax_hrv.legend()
                        self.ax_hrv.grid(True)
                except Exception as e:
                    logger.error(f"HRV 그래프 업데이트 중 오류: {e}")
                
                # 자극 신호 업데이트
                try:
                    if self.stimulator_simulator.times:
                        self.ax_stim.clear()
                        
                        # 시간 범위 (최근 5초)
                        end_time = self.stimulator_simulator.times[-1] if self.stimulator_simulator.times else 0
                        start_time = max(0, end_time - 5)
                        
                        # 표시할 데이터 인덱스 찾기
                        start_idx = 0
                        for i, t in enumerate(self.stimulator_simulator.times):
                            if t >= start_time:
                                start_idx = i
                                break
                        
                        # 각 자극기 신호 플롯
                        for i in range(self.stimulator_simulator.num_stimulators):
                            if start_idx < len(self.stimulator_simulator.generated_signals[i]):
                                self.ax_stim.plot(
                                    self.stimulator_simulator.times[start_idx:],
                                    self.stimulator_simulator.generated_signals[i][start_idx:],
                                    label=f'자극기 {i+1}'
                                )
                        
                        self.ax_stim.set_title("자극 신호")
                        self.ax_stim.set_xlabel("시간 (초)")
                        self.ax_stim.set_ylabel("진폭")
                        self.ax_stim.legend()
                        self.ax_stim.set_ylim(-1.1, 1.1)
                        self.ax_stim.grid(True)
                except Exception as e:
                    logger.error(f"자극 그래프 업데이트 중 오류: {e}")
                
                # 그래프 업데이트
                self.fig.tight_layout()
                self.canvas.draw()
                
            except Exception as e:
                logger.error(f"그래프 업데이트 중 오류: {e}")
            
            # 100ms 후 다시 호출
            self.root.after(100, self.update_plot)
    
    def start_simulation(self):
        """
        시뮬레이션 시작
        """
        if self.running:
            return
        
        self.running = True
        self.last_status_update = 0
        
        # 시나리오 설정
        scenario = self.scenario_var.get()
        self.current_scenario = scenario
        
        # 설정 적용
        self.apply_ecg_settings()
        self.apply_stimulator_settings()
        
        # 컨트롤러 시작
        self.controller.start()
        
        # 버튼 상태 변경
        if not self.headless:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.manual_stim_button.config(state=tk.NORMAL)
        
        # 시나리오 스레드 시작
        self.scenario_thread = threading.Thread(
            target=self._run_scenario,
            args=(scenario,)
        )
        self.scenario_thread.daemon = True
        self.scenario_thread.start()
        
        self.log_status(f"시뮬레이션이 시작되었습니다. 시나리오: {scenario}")
    
    def stop_simulation(self):
        """
        시뮬레이션 중지
        """
        if not self.running:
            return
        
        self.running = False
        
        # 컨트롤러 중지
        self.controller.stop()
        
        # 버튼 상태 변경
        if not self.headless:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.manual_stim_button.config(state=tk.DISABLED)
        
        self.log_status("시뮬레이션이 중지되었습니다.")
    
    def trigger_manual_stimulation(self):
        """
        수동 자극 트리거
        """
        if not self.running:
            return
        
        # 수동 자극 요청
        self.controller.manual_stimulation(
            duration=60,  # 1분 자극
            parameters={
                'frequency': self.freq_var.get(),
                'waveform_type': self.waveform_var.get(),
                'phase_delay': self.delay_var.get()
            }
        )
        
        self.log_status("수동 자극이 시작되었습니다.")
    
    def _run_scenario(self, scenario):
        """
        시나리오 실행
        
        매개변수:
            scenario (str): 시나리오 이름
        """
        try:
            if scenario == "normal":
                # 정상 상태 시나리오
                self._run_normal_scenario()
            elif scenario == "increasing_anxiety":
                # 불안 수준 증가 시나리오
                self._run_increasing_anxiety_scenario()
            elif scenario == "sudden_anxiety":
                # 갑작스런 불안 발작 시나리오
                self._run_sudden_anxiety_scenario()
            elif scenario == "recovery_after_anxiety":
                # 불안 발작 후 회복 시나리오
                self._run_recovery_after_anxiety_scenario()
            else:
                logger.warning(f"알 수 없는 시나리오: {scenario}")
                
        except Exception as e:
            logger.error(f"시나리오 실행 중 오류 발생: {e}")
    
    def _run_normal_scenario(self):
        """
        정상 상태 시나리오 실행
        
        정상적인 ECG 패턴과 HRV를 유지합니다.
        """
        self.log_status("정상 상태 시나리오를 실행합니다.")
        
        # 정상 심박 패턴 설정
        self.ecg_simulator.set_pattern(
            ECGPatternType.NORMAL,
            heart_rate=70,
            hrv_level=0.1
        )
        
        # 시나리오 지속 시간 동안 대기
        duration = 300  # 5분
        start_time = time.time()
        
        while self.running and time.time() - start_time < duration:
            # 약간의 변동 추가
            hr_variation = np.random.normal(0, 2)  # 표준편차 2bpm
            current_hr = 70 + hr_variation
            
            hrv_variation = np.random.normal(0, 0.01)  # 표준편차 0.01
            current_hrv = 0.1 + hrv_variation
            
            # 설정 업데이트
            self.ecg_simulator.set_pattern(
                ECGPatternType.NORMAL,
                heart_rate=max(60, min(80, current_hr)),
                hrv_level=max(0.08, min(0.12, current_hrv))
            )
            
            # GUI 업데이트 (헤드리스 모드가 아닌 경우)
            if not self.headless:
                self.hr_var.set(int(current_hr))
                self.hrv_var.set(current_hrv)
            
            # 30초마다 로그 기록
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                self.log_status(f"정상 상태 유지 중... (경과 시간: {int(elapsed)}초)")
            
            # 잠시 대기
            time.sleep(5)
        
        self.log_status("정상 상태 시나리오가 완료되었습니다.")
    
    def _run_increasing_anxiety_scenario(self):
        """
        불안 수준 증가 시나리오 실행
        
        심박수가 점진적으로 증가하고 HRV가 감소하여 불안 수준이 높아지는 패턴을 시뮬레이션합니다.
        """
        self.log_status("불안 수준 증가 시나리오를 실행합니다.")
        
        # 초기 설정
        initial_hr = 70
        initial_hrv = 0.1
        
        # 시나리오 지속 시간
        duration = 300  # 5분
        start_time = time.time()
        
        # 점진적 변화
        while self.running and time.time() - start_time < duration:
            # 경과 시간 비율 (0-1)
            progress = min(1.0, (time.time() - start_time) / duration)
            
            # 심박수 증가 (70bpm -> 100bpm)
            current_hr = initial_hr + progress * 30
            
            # HRV 감소 (0.1 -> 0.03)
            current_hrv = initial_hrv - progress * 0.07
            
            # 약간의 무작위성 추가
            hr_noise = np.random.normal(0, 2)  # 표준편차 2bpm
            hrv_noise = np.random.normal(0, 0.005)  # 표준편차 0.005
            
            current_hr += hr_noise
            current_hrv += hrv_noise
            
            # 값 범위 제한
            current_hr = max(65, min(105, current_hr))
            current_hrv = max(0.02, min(0.12, current_hrv))
            
            # 진행 상태에 따라 패턴 선택
            if progress < 0.3:
                pattern = ECGPatternType.NORMAL
            elif progress < 0.6:
                pattern = ECGPatternType.REDUCED_HRV
            else:
                pattern = ECGPatternType.ANXIETY
            
            # 설정 업데이트
            self.ecg_simulator.set_pattern(
                pattern,
                heart_rate=current_hr,
                hrv_level=current_hrv
            )
            
            # GUI 업데이트 (헤드리스 모드가 아닌 경우)
            if not self.headless:
                self.hr_var.set(int(current_hr))
                self.hrv_var.set(current_hrv)
                self.pattern_var.set(pattern.value)
            
            # 30초마다 로그 기록
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                self.log_status(
                    f"불안 수준 증가 중... "
                    f"(경과 시간: {int(elapsed)}초, HR: {current_hr:.1f}bpm, HRV: {current_hrv:.3f})"
                )
            
            # 잠시 대기
            time.sleep(5)
        
        self.log_status("불안 수준 증가 시나리오가 완료되었습니다.")
    
    def _run_sudden_anxiety_scenario(self):
        """
        갑작스런 불안 발작 시나리오 실행
        
        처음에는 정상 상태를 유지하다가 갑자기 불안 패턴으로 전환되는 시나리오입니다.
        """
        self.log_status("갑작스런 불안 발작 시나리오를 실행합니다.")
        
        # 시나리오 지속 시간
        duration = 300  # 5분
        start_time = time.time()
        
        # 불안 발작 시점
        attack_time = 120  # 2분 지점
        
        # 시나리오 실행
        while self.running and time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # 불안 발작 이전: 정상 상태
            if elapsed < attack_time:
                # 약간의 변동 추가
                hr_variation = np.random.normal(0, 2)
                hrv_variation = np.random.normal(0, 0.01)
                
                # 설정 업데이트
                self.ecg_simulator.set_pattern(
                    ECGPatternType.NORMAL,
                    heart_rate=70 + hr_variation,
                    hrv_level=0.1 + hrv_variation
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(int(70 + hr_variation))
                    self.hrv_var.set(0.1 + hrv_variation)
                    self.pattern_var.set(ECGPatternType.NORMAL.value)
                
                # 30초마다 로그 기록
                if int(elapsed) % 30 == 0:
                    self.log_status(f"정상 상태 유지 중... (경과 시간: {int(elapsed)}초)")
                
            # 불안 발작 시작
            elif elapsed == attack_time or (attack_time <= elapsed < attack_time + 5):
                self.log_status("갑작스런 불안 발작 시작!")
                
                # 불안 패턴으로 급격한 전환
                self.ecg_simulator.set_pattern(
                    ECGPatternType.ANXIETY,
                    heart_rate=110,
                    hrv_level=0.03
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(110)
                    self.hrv_var.set(0.03)
                    self.pattern_var.set(ECGPatternType.ANXIETY.value)
                
            # 불안 발작 진행
            else:
                # 약간의 변동 추가
                hr_variation = np.random.normal(0, 5)
                hrv_variation = np.random.normal(0, 0.005)
                
                # 설정 업데이트
                self.ecg_simulator.set_pattern(
                    ECGPatternType.ANXIETY,
                    heart_rate=110 + hr_variation,
                    hrv_level=0.03 + hrv_variation
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(int(110 + hr_variation))
                    self.hrv_var.set(0.03 + hrv_variation)
                
                # 30초마다 로그 기록
                if int(elapsed) % 30 == 0:
                    self.log_status(
                        f"불안 발작 진행 중... "
                        f"(경과 시간: {int(elapsed)}초, HR: {110+hr_variation:.1f}bpm, HRV: {0.03+hrv_variation:.3f})"
                    )
            
            # 잠시 대기
            time.sleep(5)
        
        self.log_status("갑작스런 불안 발작 시나리오가 완료되었습니다.")
    
    def _run_recovery_after_anxiety_scenario(self):
        """
        불안 발작 후 회복 시나리오 실행
        
        불안 상태가 일정 시간 지속된 후 (자극 이후) 서서히 정상 상태로 회복되는 시나리오입니다.
        """
        self.log_status("불안 발작 후 회복 시나리오를 실행합니다.")
        
        # 시나리오 지속 시간
        duration = 360  # 6분
        start_time = time.time()
        
        # 단계별 지속 시간
        anxiety_duration = 90   # 불안 상태 (1분 30초)
        stim_duration = 60      # 자극 시간 (1분)
        recovery_duration = 210  # 회복 시간 (3분 30초)
        
        # 현재 단계
        current_stage = "anxiety"
        
        # 시나리오 실행
        while self.running and time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # 단계 결정
            if elapsed < anxiety_duration:
                current_stage = "anxiety"
            elif elapsed < anxiety_duration + stim_duration:
                current_stage = "stimulation"
            else:
                current_stage = "recovery"
                # 회복 진행 상태 (0-1)
                recovery_progress = min(1.0, (elapsed - (anxiety_duration + stim_duration)) / recovery_duration)
            
            # 단계별 처리
            if current_stage == "anxiety":
                # 불안 상태 유지
                hr_variation = np.random.normal(0, 3)
                hrv_variation = np.random.normal(0, 0.005)
                
                # 설정 업데이트
                self.ecg_simulator.set_pattern(
                    ECGPatternType.ANXIETY,
                    heart_rate=100 + hr_variation,
                    hrv_level=0.04 + hrv_variation
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(int(100 + hr_variation))
                    self.hrv_var.set(0.04 + hrv_variation)
                    self.pattern_var.set(ECGPatternType.ANXIETY.value)
                
                # 로깅
                if elapsed < 5 or int(elapsed) % 30 == 0:
                    self.log_status(
                        f"불안 상태 유지 중... "
                        f"(경과 시간: {int(elapsed)}초, HR: {100+hr_variation:.1f}bpm, HRV: {0.04+hrv_variation:.3f})"
                    )
                
            elif current_stage == "stimulation":
                # 자극 시작 시점에 수동 자극 트리거
                if abs(elapsed - anxiety_duration) < 5:
                    self.log_status("자극 시작...")
                    
                    # 자극 파라미터 설정
                    self.controller.set_stimulation_parameters({
                        'frequency': 15.0,
                        'phase_delay': 0.5,
                        'waveform_type': 'sine',
                        'phase_type': 'biphasic',
                        'duration': stim_duration
                    })
                    
                    # 수동 자극 시작
                    self.controller.manual_stimulation(duration=stim_duration)
                    
                    # GUI 업데이트
                    if not self.headless:
                        self.freq_var.set(15.0)
                        self.delay_var.set(0.5)
                        self.waveform_var.set('sine')
                
                # 자극 중 불안 수준 서서히 감소
                stim_progress = (elapsed - anxiety_duration) / stim_duration
                
                # 심박수: 100 -> 90
                current_hr = 100 - stim_progress * 10 + np.random.normal(0, 2)
                
                # HRV: 0.04 -> 0.06
                current_hrv = 0.04 + stim_progress * 0.02 + np.random.normal(0, 0.005)
                
                # 설정 업데이트
                self.ecg_simulator.set_pattern(
                    ECGPatternType.REDUCED_HRV,
                    heart_rate=current_hr,
                    hrv_level=current_hrv
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(int(current_hr))
                    self.hrv_var.set(current_hrv)
                    self.pattern_var.set(ECGPatternType.REDUCED_HRV.value)
                
                # 로깅
                if int(elapsed) % 20 == 0:
                    self.log_status(
                        f"자극 중, 불안 수준 감소 중... "
                        f"(경과 시간: {int(elapsed)}초, HR: {current_hr:.1f}bpm, HRV: {current_hrv:.3f})"
                    )
                
            elif current_stage == "recovery":
                # 서서히 정상 상태로 회복
                
                # 심박수: 90 -> 70
                current_hr = 90 - recovery_progress * 20 + np.random.normal(0, 2)
                
                # HRV: 0.06 -> 0.1
                current_hrv = 0.06 + recovery_progress * 0.04 + np.random.normal(0, 0.005)
                
                # 패턴 변화: 회복 진행에 따라 패턴 전환
                if recovery_progress < 0.3:
                    pattern = ECGPatternType.REDUCED_HRV
                elif recovery_progress < 0.7:
                    pattern = ECGPatternType.ELEVATED_HR
                else:
                    pattern = ECGPatternType.NORMAL
                
                # 설정 업데이트
                self.ecg_simulator.set_pattern(
                    pattern,
                    heart_rate=current_hr,
                    hrv_level=current_hrv
                )
                
                # GUI 업데이트
                if not self.headless:
                    self.hr_var.set(int(current_hr))
                    self.hrv_var.set(current_hrv)
                    self.pattern_var.set(pattern.value)
                
                # 로깅
                if int(elapsed) % 30 == 0:
                    self.log_status(
                        f"회복 중... "
                        f"(경과 시간: {int(elapsed)}초, HR: {current_hr:.1f}bpm, HRV: {current_hrv:.3f}, 진행: {recovery_progress*100:.0f}%)"
                    )
            
            # 잠시 대기
            time.sleep(5)
        
        self.log_status("불안 발작 후 회복 시나리오가 완료되었습니다.")


def main():
    """
    시뮬레이션 실행 메인 함수
    """
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='불안장애 예방 시스템 시뮬레이션')
    parser.add_argument('--headless', action='store_true', help='헤드리스 모드 (GUI 없음)')
    parser.add_argument('--scenario', type=str, default='normal', 
                        choices=['normal', 'increasing_anxiety', 'sudden_anxiety', 'recovery_after_anxiety'],
                        help='실행할 시나리오')
    parser.add_argument('--duration', type=int, default=300, help='시뮬레이션 지속 시간 (초)')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='로그 레벨')
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 헤드리스 모드
    if args.headless:
        # GUI 없이 시뮬레이션 실행
        app = SimulationApp(headless=True)
        
        # 시나리오 설정 및 시작
        app.scenario_var = args.scenario
        app.start_simulation()
        
        try:
            # 지정된 시간 동안 실행
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
        finally:
            # 종료
            app.stop_simulation()
            
    else:
        # GUI 모드
        root = tk.Tk()
        app = SimulationApp(root=root)
        
        # 종료 시 정리
        def on_closing():
            if app.running:
                app.stop_simulation()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()


if __name__ == "__main__":
    main()
