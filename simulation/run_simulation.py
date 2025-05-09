#!/usr/bin/env python3
"""
Anxiety Prevention Device - Integrated Simulation Environment

This script provides a complete simulation environment for the anxiety prevention system
based on the patents 10-2022-0007209 ("불안장애 예방장치") and 10-2459338 ("저주파 자극기 제어장치").

The simulation integrates:
1. ECG signal generation and HRV measurement
2. Real-time HRV analysis and anxiety prediction
3. Cranial electrotherapy stimulation control with multiple devices
4. Visual interface showing the complete workflow

Usage:
    python run_simulation.py [--headless]
    
    --headless: Run in headless mode (no GUI)
"""

import os
import sys
import time
import argparse
import threading
import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import simulation modules
from simulation.ecg_simulator import ECGSimulator
from simulation.stimulator_simulator import StimulatorSimulator

# Import core modules
from src.hrv_analyzer.hrv_analysis import HRVAnalyzer
from src.hrv_analyzer.anxiety_predictor import AnxietyPredictor
from src.stimulation_controller.cranial_stimulator import CranialStimulator, WaveformType

# GUI imports
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    import pyqtgraph as pg
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)
logger = logging.getLogger(__name__)

class SimulationController:
    """
    Main controller for the anxiety prevention device simulation.
    
    This class integrates all components of the system and manages the simulation flow.
    """
    
    def __init__(self, headless: bool = False):
        """
        Initialize the simulation controller.
        
        Args:
            headless: Whether to run in headless mode (no GUI).
        """
        self.headless = headless
        self.running = False
        self.pause = False
        
        # Initialize simulation components
        self.ecg_simulator = ECGSimulator()
        self.hrv_analyzer = HRVAnalyzer()
        self.anxiety_predictor = AnxietyPredictor()
        self.cranial_stimulator = CranialStimulator()
        
        # Add stimulator devices
        self.stimulator1 = StimulatorSimulator(device_id="STIM001", name="Left Stimulator")
        self.stimulator2 = StimulatorSimulator(device_id="STIM002", name="Right Stimulator")
        
        self.cranial_stimulator.add_device("STIM001", self.stimulator1)
        self.cranial_stimulator.add_device("STIM002", self.stimulator2)
        
        # Connect to stimulators
        self.cranial_stimulator.connect_device("STIM001")
        self.cranial_stimulator.connect_device("STIM002")
        
        # Initialize timers
        self.ecg_timer = None
        self.hrv_timer = None
        self.anxiety_timer = None
        
        # Initialize data storage
        self.ecg_data = []
        self.hrv_data = []
        self.anxiety_data = []
        
        # Initialize simulation parameters
        self.simulation_speed = 1.0  # Real-time simulation by default
        self.current_anxiety_level = 0
        
        # Initialize GUI if not in headless mode
        self.app = None
        self.main_window = None
        
        if not headless and GUI_AVAILABLE:
            self._setup_gui()
    
    def _setup_gui(self):
        """Set up the graphical user interface."""
        self.app = QtWidgets.QApplication([])
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("Anxiety Prevention Device Simulation")
        self.main_window.resize(1200, 800)
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.main_window.setCentralWidget(central_widget)
        
        # Add control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Start/Stop button
        self.start_stop_button = QtWidgets.QPushButton("Start Simulation")
        self.start_stop_button.clicked.connect(self._toggle_simulation)
        control_panel.addWidget(self.start_stop_button)
        
        # Pause/Resume button
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.pause_button.setEnabled(False)
        control_panel.addWidget(self.pause_button)
        
        # Speed control
        control_panel.addWidget(QtWidgets.QLabel("Simulation Speed:"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(500)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self._update_simulation_speed)
        control_panel.addWidget(self.speed_slider)
        
        # Speed label
        self.speed_label = QtWidgets.QLabel("1.0x")
        control_panel.addWidget(self.speed_label)
        
        # Anxiety level control
        control_panel.addWidget(QtWidgets.QLabel("Simulated Anxiety:"))
        self.anxiety_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.anxiety_slider.setMinimum(0)
        self.anxiety_slider.setMaximum(100)
        self.anxiety_slider.setValue(20)
        self.anxiety_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.anxiety_slider.setTickInterval(10)
        self.anxiety_slider.valueChanged.connect(self._update_simulated_anxiety)
        control_panel.addWidget(self.anxiety_slider)
        
        main_layout.addLayout(control_panel)
        
        # Create tab widget for different views
        tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # ECG and HRV tab
        ecg_hrv_tab = QtWidgets.QWidget()
        ecg_hrv_layout = QtWidgets.QVBoxLayout()
        ecg_hrv_tab.setLayout(ecg_hrv_layout)
        
        # ECG plot
        ecg_plot_widget = pg.PlotWidget(title="ECG Signal")
        ecg_plot_widget.setLabel('left', "Amplitude", units='mV')
        ecg_plot_widget.setLabel('bottom', "Time", units='s')
        ecg_plot_widget.showGrid(x=True, y=True)
        ecg_plot_widget.setYRange(-1.5, 1.5)
        self.ecg_curve = ecg_plot_widget.plot(pen='g')
        ecg_hrv_layout.addWidget(ecg_plot_widget)
        
        # HRV plots
        hrv_plots_layout = QtWidgets.QHBoxLayout()
        
        # Time domain HRV
        time_hrv_plot = pg.PlotWidget(title="HRV Time Domain (RMSSD)")
        time_hrv_plot.setLabel('left', "RMSSD", units='ms')
        time_hrv_plot.setLabel('bottom', "Time", units='min')
        time_hrv_plot.showGrid(x=True, y=True)
        self.time_hrv_curve = time_hrv_plot.plot(pen='b')
        hrv_plots_layout.addWidget(time_hrv_plot)
        
        # Frequency domain HRV
        freq_hrv_plot = pg.PlotWidget(title="HRV Frequency Domain (LF/HF Ratio)")
        freq_hrv_plot.setLabel('left', "LF/HF Ratio")
        freq_hrv_plot.setLabel('bottom', "Time", units='min')
        freq_hrv_plot.showGrid(x=True, y=True)
        self.freq_hrv_curve = freq_hrv_plot.plot(pen='r')
        hrv_plots_layout.addWidget(freq_hrv_plot)
        
        ecg_hrv_layout.addLayout(hrv_plots_layout)
        
        # Anxiety analysis tab
        anxiety_tab = QtWidgets.QWidget()
        anxiety_layout = QtWidgets.QVBoxLayout()
        anxiety_tab.setLayout(anxiety_layout)
        
        # Anxiety prediction plot
        anxiety_plot = pg.PlotWidget(title="Anxiety Prediction")
        anxiety_plot.setLabel('left', "Anxiety Level")
        anxiety_plot.setLabel('bottom', "Time", units='min')
        anxiety_plot.showGrid(x=True, y=True)
        anxiety_plot.setYRange(-0.1, 3.1)
        anxiety_plot.addLine(y=0.5, pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashLine))
        anxiety_plot.addLine(y=1.5, pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        anxiety_plot.addLine(y=2.5, pen=pg.mkPen('r', width=1, style=QtCore.Qt.DashLine))
        self.anxiety_curve = anxiety_plot.plot(pen='c', symbolBrush=(255,0,0), symbolPen='w')
        
        # Add legend
        legend = pg.LegendItem(offset=(70, 30))
        legend.setParentItem(anxiety_plot.graphicsItem())
        legend.addItem(self.anxiety_curve, "Anxiety Level")
        
        anxiety_layout.addWidget(anxiety_plot)
        
        # Anxiety indicators
        indicators_layout = QtWidgets.QHBoxLayout()
        
        # Current anxiety level
        anxiety_info_layout = QtWidgets.QVBoxLayout()
        anxiety_info_group = QtWidgets.QGroupBox("Current Anxiety Status")
        anxiety_info_inner_layout = QtWidgets.QFormLayout()
        self.anxiety_level_label = QtWidgets.QLabel("0")
        self.anxiety_category_label = QtWidgets.QLabel("Low")
        self.anxiety_confidence_label = QtWidgets.QLabel("0.00")
        
        anxiety_info_inner_layout.addRow("Level (0-3):", self.anxiety_level_label)
        anxiety_info_inner_layout.addRow("Category:", self.anxiety_category_label)
        anxiety_info_inner_layout.addRow("Confidence:", self.anxiety_confidence_label)
        
        anxiety_info_group.setLayout(anxiety_info_inner_layout)
        anxiety_info_layout.addWidget(anxiety_info_group)
        indicators_layout.addLayout(anxiety_info_layout)
        
        # Visual anxiety indicator
        self.anxiety_indicator = QtWidgets.QProgressBar()
        self.anxiety_indicator.setOrientation(QtCore.Qt.Vertical)
        self.anxiety_indicator.setMinimum(0)
        self.anxiety_indicator.setMaximum(100)
        self.anxiety_indicator.setValue(0)
        self.anxiety_indicator.setTextVisible(False)
        self.anxiety_indicator.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #00AA00;
            }
        """)
        indicators_layout.addWidget(self.anxiety_indicator)
        
        anxiety_layout.addLayout(indicators_layout)
        
        # Stimulation tab
        stimulation_tab = QtWidgets.QWidget()
        stimulation_layout = QtWidgets.QVBoxLayout()
        stimulation_tab.setLayout(stimulation_layout)
        
        # Stimulation controls
        stim_controls_layout = QtWidgets.QHBoxLayout()
        
        # Stimulation status
        stim_status_group = QtWidgets.QGroupBox("Stimulation Status")
        stim_status_layout = QtWidgets.QFormLayout()
        self.stim_active_label = QtWidgets.QLabel("Not Active")
        self.stim_duration_label = QtWidgets.QLabel("0:00")
        self.stim_remaining_label = QtWidgets.QLabel("0:00")
        
        stim_status_layout.addRow("Status:", self.stim_active_label)
        stim_status_layout.addRow("Duration:", self.stim_duration_label)
        stim_status_layout.addRow("Remaining:", self.stim_remaining_label)
        
        stim_status_group.setLayout(stim_status_layout)
        stim_controls_layout.addWidget(stim_status_group)
        
        # Stimulation parameters
        stim_params_group = QtWidgets.QGroupBox("Stimulation Parameters")
        stim_params_layout = QtWidgets.QFormLayout()
        self.stim_waveform_label = QtWidgets.QLabel("Sine")
        self.stim_frequency_label = QtWidgets.QLabel("0.5 Hz")
        self.stim_phase_diff_label = QtWidgets.QLabel("0.5 s")
        
        stim_params_layout.addRow("Waveform:", self.stim_waveform_label)
        stim_params_layout.addRow("Frequency:", self.stim_frequency_label)
        stim_params_layout.addRow("Phase Difference:", self.stim_phase_diff_label)
        
        stim_params_group.setLayout(stim_params_layout)
        stim_controls_layout.addWidget(stim_params_group)
        
        # Manual stimulation controls
        stim_manual_group = QtWidgets.QGroupBox("Manual Control")
        stim_manual_layout = QtWidgets.QVBoxLayout()
        
        # Stimulation device selection
        self.stim1_checkbox = QtWidgets.QCheckBox("Left Stimulator (STIM001)")
        self.stim1_checkbox.setChecked(True)
        self.stim2_checkbox = QtWidgets.QCheckBox("Right Stimulator (STIM002)")
        self.stim2_checkbox.setChecked(True)
        
        # Waveform selection
        waveform_layout = QtWidgets.QHBoxLayout()
        waveform_layout.addWidget(QtWidgets.QLabel("Waveform:"))
        self.waveform_combo = QtWidgets.QComboBox()
        self.waveform_combo.addItems([w.value for w in WaveformType])
        waveform_layout.addWidget(self.waveform_combo)
        
        # Stimulation buttons
        stim_buttons_layout = QtWidgets.QHBoxLayout()
        self.start_stim_button = QtWidgets.QPushButton("Start Manual Stimulation")
        self.start_stim_button.clicked.connect(self._start_manual_stimulation)
        self.stop_stim_button = QtWidgets.QPushButton("Stop Stimulation")
        self.stop_stim_button.clicked.connect(self._stop_stimulation)
        self.stop_stim_button.setEnabled(False)
        
        stim_buttons_layout.addWidget(self.start_stim_button)
        stim_buttons_layout.addWidget(self.stop_stim_button)
        
        stim_manual_layout.addWidget(self.stim1_checkbox)
        stim_manual_layout.addWidget(self.stim2_checkbox)
        stim_manual_layout.addLayout(waveform_layout)
        stim_manual_layout.addLayout(stim_buttons_layout)
        
        stim_manual_group.setLayout(stim_manual_layout)
        stim_controls_layout.addWidget(stim_manual_group)
        
        stimulation_layout.addLayout(stim_controls_layout)
        
        # Stimulation visualization
        stim_viz_layout = QtWidgets.QHBoxLayout()
        
        # Device 1 waveform
        stim1_plot = pg.PlotWidget(title="Left Stimulator Waveform")
        stim1_plot.setLabel('left', "Amplitude")
        stim1_plot.setLabel('bottom', "Time", units='s')
        stim1_plot.showGrid(x=True, y=True)
        stim1_plot.setYRange(-1.1, 1.1)
        self.stim1_curve = stim1_plot.plot(pen='b')
        stim_viz_layout.addWidget(stim1_plot)
        
        # Device 2 waveform
        stim2_plot = pg.PlotWidget(title="Right Stimulator Waveform")
        stim2_plot.setLabel('left', "Amplitude")
        stim2_plot.setLabel('bottom', "Time", units='s')
        stim2_plot.showGrid(x=True, y=True)
        stim2_plot.setYRange(-1.1, 1.1)
        self.stim2_curve = stim2_plot.plot(pen='r')
        stim_viz_layout.addWidget(stim2_plot)
        
        stimulation_layout.addLayout(stim_viz_layout)
        
        # Add tabs to tab widget
        tab_widget.addTab(ecg_hrv_tab, "ECG & HRV Analysis")
        tab_widget.addTab(anxiety_tab, "Anxiety Prediction")
        tab_widget.addTab(stimulation_tab, "CES Stimulation")
        
        # Add log widget
        log_group = QtWidgets.QGroupBox("Simulation Log")
        log_layout = QtWidgets.QVBoxLayout()
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        log_layout.addWidget(self.log_widget)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Setup log handler
        class QTextEditLogger(logging.Handler):
            def __init__(self, widget):
                super().__init__()
                self.widget = widget
                self.widget.setStyleSheet("font-family: monospace")
                self.widget.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
                self.widget.document().setMaximumBlockCount(500)  # Limit to 500 lines
                
            def emit(self, record):
                msg = self.format(record)
                self.widget.append(msg)
        
        log_handler = QTextEditLogger(self.log_widget)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        
        # Setup update timers
        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self._update_gui)
        self.gui_timer.start(50)  # Update every 50ms
        
        # Show the main window
        self.main_window.show()
    
    def _toggle_simulation(self):
        """Toggle the simulation between running and stopped states."""
        if not self.running:
            self.start_simulation()
            if hasattr(self, 'start_stop_button'):
                self.start_stop_button.setText("Stop Simulation")
                self.pause_button.setEnabled(True)
        else:
            self.stop_simulation()
            if hasattr(self, 'start_stop_button'):
                self.start_stop_button.setText("Start Simulation")
                self.pause_button.setEnabled(False)
                self.pause_button.setText("Pause")
    
    def _toggle_pause(self):
        """Toggle the simulation between paused and running states."""
        self.pause = not self.pause
        if self.pause:
            if hasattr(self, 'pause_button'):
                self.pause_button.setText("Resume")
            logger.info("Simulation paused")
        else:
            if hasattr(self, 'pause_button'):
                self.pause_button.setText("Pause")
            logger.info("Simulation resumed")
    
    def _update_simulation_speed(self):
        """Update the simulation speed based on slider value."""
        speed_value = self.speed_slider.value() / 100.0
        self.simulation_speed = speed_value
        self.speed_label.setText(f"{speed_value:.1f}x")
    
    def _update_simulated_anxiety(self):
        """Update the simulated anxiety level based on slider value."""
        anxiety_value = self.anxiety_slider.value() / 100.0 * 3.0  # Scale to 0-3 range
        # This will be used when generating simulated ECG
    
    def _start_manual_stimulation(self):
        """Start manual stimulation based on GUI settings."""
        if self.cranial_stimulator.is_stimulating:
            logger.warning("Stimulation is already active")
            return
        
        # Get selected devices
        devices = []
        if self.stim1_checkbox.isChecked():
            devices.append("STIM001")
        if self.stim2_checkbox.isChecked():
            devices.append("STIM002")
        
        if not devices:
            logger.warning("No stimulation devices selected")
            return
        
        # Get selected waveform
        waveform_str = self.waveform_combo.currentText()
        waveform = next((w for w in WaveformType if w.value == waveform_str), WaveformType.SINE)
        
        # Start stimulation
        intensities = {device_id: 0.5 for device_id in devices}
        success = self.cranial_stimulator.start_stimulation(
            anxiety_level=self.current_anxiety_level,
            session_duration=5,  # 5 minutes for manual stimulation
            waveform=waveform,
            base_frequency=0.5,
            phase_difference=0.5,
            intensities=intensities
        )
        
        if success:
            logger.info(f"Started manual stimulation with {len(devices)} device(s)")
            self.start_stim_button.setEnabled(False)
            self.stop_stim_button.setEnabled(True)
        else:
            logger.error("Failed to start manual stimulation")
    
    def _stop_stimulation(self):
        """Stop any active stimulation."""
        if not self.cranial_stimulator.is_stimulating:
            logger.warning("No active stimulation to stop")
            return
        
        success = self.cranial_stimulator.stop_stimulation()
        
        if success:
            logger.info("Stopped stimulation")
            self.start_stim_button.setEnabled(True)
            self.stop_stim_button.setEnabled(False)
        else:
            logger.error("Failed to stop stimulation")
    
    def _update_gui(self):
        """Update the GUI with the latest simulation data."""
        if not self.running:
            return
        
        # Update ECG plot
        if self.ecg_data:
            # Show last 10 seconds of ECG data
            sample_rate = self.ecg_simulator.sample_rate
            window_size = min(len(self.ecg_data), 10 * sample_rate)
            recent_ecg = self.ecg_data[-window_size:]
            time_axis = np.linspace(0, len(recent_ecg) / sample_rate, len(recent_ecg))
            self.ecg_curve.setData(time_axis, recent_ecg)
        
        # Update HRV plots
        if self.hrv_data:
            time_points = np.arange(len(self.hrv_data)) / 60.0  # Convert to minutes
            
            # Extract RMSSD values
            rmssd_values = [data['RMSSD'] for data in self.hrv_data]
            self.time_hrv_curve.setData(time_points, rmssd_values)
            
            # Extract LF/HF ratio values
            lf_hf_values = [data['LF_HF_ratio'] for data in self.hrv_data]
            self.freq_hrv_curve.setData(time_points, lf_hf_values)
        
        # Update anxiety plot
        if self.anxiety_data:
            time_points = np.arange(len(self.anxiety_data)) / 60.0  # Convert to minutes
            anxiety_levels = [data['risk_level'] for data in self.anxiety_data]
            self.anxiety_curve.setData(time_points, anxiety_levels)
            
            # Update current anxiety status
            if self.anxiety_data:
                latest = self.anxiety_data[-1]
                self.anxiety_level_label.setText(str(latest['risk_level']))
                self.anxiety_category_label.setText(latest['risk_category'])
                self.anxiety_confidence_label.setText(f"{latest['confidence']:.2f}")
                
                # Update anxiety indicator
                indicator_value = int(latest['risk_level'] * 100 / 3)  # Scale to 0-100
                self.anxiety_indicator.setValue(indicator_value)
                
                # Change indicator color based on anxiety level
                if latest['risk_level'] == 0:
                    color = "#00AA00"  # Green
                elif latest['risk_level'] == 1:
                    color = "#AAAA00"  # Yellow
                elif latest['risk_level'] == 2:
                    color = "#FF7700"  # Orange
                else:
                    color = "#FF0000"  # Red
                
                self.anxiety_indicator.setStyleSheet(f"""
                    QProgressBar {{
                        border: 2px solid grey;
                        border-radius: 5px;
                        background-color: #FFFFFF;
                    }}
                    QProgressBar::chunk {{
                        background-color: {color};
                    }}
                """)
        
        # Update stimulation status
        stim_status = self.cranial_stimulator.get_stimulation_status()
        if stim_status['is_stimulating']:
            self.stim_active_label.setText("Active")
            
            session = stim_status['session']
            if session:
                # Format duration
                duration_mins = int(session['duration'] / 60)
                duration_secs = int(session['duration'] % 60)
                self.stim_duration_label.setText(f"{duration_mins}:{duration_secs:02d}")
                
                # Format remaining time
                remaining_mins = int(session['remaining_time'] / 60)
                remaining_secs = int(session['remaining_time'] % 60)
                self.stim_remaining_label.setText(f"{remaining_mins}:{remaining_secs:02d}")
                
                # Update parameter labels
                self.stim_waveform_label.setText(session['waveform'])
                self.stim_frequency_label.setText(f"{self.cranial_stimulator.current_session['base_frequency']:.1f} Hz")
                self.stim_phase_diff_label.setText(f"{self.cranial_stimulator.current_session['phase_difference']:.1f} s")
                
                # Update waveform plots
                t = np.linspace(0, 2, 200)
                freq = self.cranial_stimulator.current_session['base_frequency']
                
                # Generate waveforms based on type
                if self.cranial_stimulator.current_session['waveform'] == WaveformType.SINE:
                    wave1 = np.sin(2 * np.pi * freq * t)
                    wave2 = np.sin(2 * np.pi * freq * (t - self.cranial_stimulator.current_session['phase_difference']))
                elif self.cranial_stimulator.current_session['waveform'] == WaveformType.SQUARE:
                    wave1 = np.sign(np.sin(2 * np.pi * freq * t))
                    wave2 = np.sign(np.sin(2 * np.pi * freq * (t - self.cranial_stimulator.current_session['phase_difference'])))
                elif self.cranial_stimulator.current_session['waveform'] == WaveformType.TRIANGULAR:
                    wave1 = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
                    wave2 = 2 * np.abs(2 * ((t - self.cranial_stimulator.current_session['phase_difference']) * freq - 
                                           np.floor((t - self.cranial_stimulator.current_session['phase_difference']) * freq + 0.5))) - 1
                elif self.cranial_stimulator.current_session['waveform'] == WaveformType.MONOPHASIC:
                    wave1 = np.maximum(0, np.sin(2 * np.pi * freq * t))
                    wave2 = np.maximum(0, np.sin(2 * np.pi * freq * (t - self.cranial_stimulator.current_session['phase_difference'])))
                else:  # BIPHASIC
                    wave1 = np.sin(2 * np.pi * freq * t)
                    wave2 = np.sin(2 * np.pi * freq * (t - self.cranial_stimulator.current_session['phase_difference']))
                
                # Scale by intensity
                if "STIM001" in self.cranial_stimulator.current_session['intensities']:
                    wave1 *= self.cranial_stimulator.current_session['intensities']["STIM001"]
                if "STIM002" in self.cranial_stimulator.current_session['intensities']:
                    wave2 *= self.cranial_stimulator.current_session['intensities']["STIM002"]
                
                self.stim1_curve.setData(t, wave1)
                self.stim2_curve.setData(t, wave2)
                
        else:
            self.stim_active_label.setText("Not Active")
            self.stim_duration_label.setText("0:00")
            self.stim_remaining_label.setText("0:00")
            
            # Clear waveform plots
            self.stim1_curve.setData([], [])
            self.stim2_curve.setData([], [])
    
    def start_simulation(self):
        """Start the simulation."""
        if self.running:
            logger.warning("Simulation is already running")
            return
        
        logger.info("Starting simulation")
        self.running = True
        self.pause = False
        
        # Reset data
        self.ecg_data = []
        self.hrv_data = []
        self.anxiety_data = []
        
        # Start ECG simulation
        self.ecg_timer = threading.Timer(0.1, self._ecg_update_task)
        self.ecg_timer.daemon = True
        self.ecg_timer.start()
        
        # Start HRV analysis
        self.hrv_timer = threading.Timer(1.0, self._hrv_update_task)
        self.hrv_timer.daemon = True
        self.hrv_timer.start()
        
        # Start anxiety prediction
        self.anxiety_timer = threading.Timer(5.0, self._anxiety_update_task)
        self.anxiety_timer.daemon = True
        self.anxiety_timer.start()
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.running:
            logger.warning("Simulation is not running")
            return
        
        logger.info("Stopping simulation")
        self.running = False
        
        # Stop any active stimulation
        if self.cranial_stimulator.is_stimulating:
            self.cranial_stimulator.stop_stimulation()
        
        # Wait for timers to finish
        # No need to join or cancel timers as they will exit naturally due to self.running = False
    
    def _ecg_update_task(self):
        """Task to periodically update simulated ECG data."""
        while self.running:
            if not self.pause:
                # Get anxiety level from slider if GUI is available
                if hasattr(self, 'anxiety_slider'):
                    anxiety_factor = self.anxiety_slider.value() / 100.0
                else:
                    # Simulate varying anxiety for headless mode
                    t = time.time() / 100.0
                    anxiety_factor = (np.sin(t) + 1) / 2.0  # Oscillate between 0 and 1
                
                # Generate ECG data with adjustments based on anxiety level
                # Higher anxiety leads to higher heart rate and lower HRV
                heart_rate = 60 + (40 * anxiety_factor)  # 60-100 BPM
                hrv_factor = 1.0 - (0.8 * anxiety_factor)  # 1.0-0.2 (less HRV with more anxiety)
                
                new_ecg_data = self.ecg_simulator.generate_ecg_segment(
                    duration=0.2,
                    heart_rate=heart_rate,
                    hrv_factor=hrv_factor,
                    noise_level=0.05 + (0.1 * anxiety_factor)
                )
                
                self.ecg_data.extend(new_ecg_data)
                
                # Keep only last 60 seconds of data
                max_samples = 60 * self.ecg_simulator.sample_rate
                if len(self.ecg_data) > max_samples:
                    self.ecg_data = self.ecg_data[-max_samples:]
            
            # Sleep interval adjusted by simulation speed
            time.sleep(0.1 / self.simulation_speed)
    
    def _hrv_update_task(self):
        """Task to periodically analyze HRV from ECG data."""
        while self.running:
            if not self.pause and len(self.ecg_data) >= 10 * self.ecg_simulator.sample_rate:
                # Use last 30 seconds of ECG data for HRV analysis
                window_size = min(len(self.ecg_data), 30 * self.ecg_simulator.sample_rate)
                ecg_segment = self.ecg_data[-window_size:]
                
                try:
                    # Analyze HRV
                    hrv_params = self.hrv_analyzer.analyze_ecg(
                        ecg_signal=np.array(ecg_segment),
                        sample_rate=self.ecg_simulator.sample_rate
                    )
                    
                    if hrv_params:
                        self.hrv_data.append(hrv_params)
                        logger.debug(f"HRV parameters updated: RMSSD={hrv_params['RMSSD']:.2f}, "
                                    f"LF/HF={hrv_params['LF_HF_ratio']:.2f}")
                except Exception as e:
                    logger.error(f"Error analyzing HRV: {str(e)}")
            
            # Sleep interval adjusted by simulation speed
            time.sleep(1.0 / self.simulation_speed)
    
    def _anxiety_update_task(self):
        """Task to periodically predict anxiety levels from HRV data."""
        intervention_cooldown = 0
        
        while self.running:
            if not self.pause and len(self.hrv_data) > 0:
                # Use the latest HRV parameters for anxiety prediction
                latest_hrv = self.hrv_data[-1]
                
                try:
                    # Predict anxiety level
                    prediction = self.anxiety_predictor.predict(latest_hrv)
                    self.anxiety_data.append(prediction)
                    self.current_anxiety_level = prediction['risk_level']
                    
                    logger.info(f"Anxiety prediction: Level {prediction['risk_level']} "
                               f"({prediction['risk_category']}) with confidence {prediction['confidence']:.2f}")
                    
                    # Check if intervention is needed
                    if (prediction['risk_level'] >= 2 and not self.cranial_stimulator.is_stimulating
                            and intervention_cooldown <= 0):
                        # Get recommended stimulation parameters
                        params = self.cranial_stimulator.get_recommended_parameters(prediction['risk_level'])
                        
                        # Start stimulation with both devices
                        success = self.cranial_stimulator.start_stimulation(
                            anxiety_level=prediction['risk_level'],
                            session_duration=params['duration'],
                            waveform=WaveformType(params['waveform']),
                            base_frequency=params['frequency'],
                            phase_difference=params['phase_difference']
                        )
                        
                        if success:
                            logger.info(f"Started automatic stimulation for anxiety level {prediction['risk_level']}")
                            intervention_cooldown = 60  # Set cooldown to 60 seconds
                            
                            if hasattr(self, 'start_stim_button') and hasattr(self, 'stop_stim_button'):
                                self.start_stim_button.setEnabled(False)
                                self.stop_stim_button.setEnabled(True)
                        else:
                            logger.error("Failed to start automatic stimulation")
                
                except Exception as e:
                    logger.error(f"Error predicting anxiety: {str(e)}")
                
                # Update cooldown
                if intervention_cooldown > 0:
                    intervention_cooldown -= 5 * self.simulation_speed
            
            # Sleep interval adjusted by simulation speed
            time.sleep(5.0 / self.simulation_speed)
    
    def run(self):
        """Run the simulation."""
        if not self.headless and GUI_AVAILABLE:
            # Start the GUI event loop
            self.app.exec_()
        else:
            # Run in headless mode
            self.start_simulation()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping simulation")
                self.stop_simulation()


def main():
    parser = argparse.ArgumentParser(description='Anxiety Prevention Device Simulation')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    args = parser.parse_args()
    
    # Check if GUI is available when not in headless mode
    if not args.headless and not GUI_AVAILABLE:
        logger.warning("GUI dependencies (PyQt5, pyqtgraph) not available. Running in headless mode.")
        args.headless = True
    
    # Create and run the simulation
    simulation = SimulationController(headless=args.headless)
    simulation.run()


if __name__ == '__main__':
    main()
