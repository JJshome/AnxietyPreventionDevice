"""
Cranial Electrotherapy Stimulation (CES) Controller

This module implements the cranial electrotherapy stimulation functionality
as described in the patent 10-2022-0007209 ("불안장애 예방장치").

It provides interfaces to control multiple stimulation devices with different 
phase signals to create a stereoscopic stimulation effect, as described in 
patent 10-2459338 ("저주파 자극기 제어장치").
"""

import time
import logging
import threading
import math
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Callable

from .stimulator_interface import StimulatorInterface

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaveformType(Enum):
    """Enum defining different types of stimulation waveforms."""
    SINE = "sine"
    SQUARE = "square"
    TRIANGULAR = "triangular"
    MONOPHASIC = "monophasic"
    BIPHASIC = "biphasic"

class CranialStimulator:
    """
    Controller for cranial electrotherapy stimulation (CES) devices.
    
    This class implements the functionality to control multiple stimulation devices
    with different phase signals to create a stereoscopic stimulation effect, as per
    the patent specifications.
    """
    
    def __init__(self):
        """Initialize the cranial stimulator controller."""
        self.devices = {}  # Dictionary to store connected stimulation devices
        self.is_stimulating = False
        self.current_session = None
        self.stimulation_thread = None
        self.stop_event = threading.Event()
    
    def add_device(self, device_id: str, device: StimulatorInterface) -> bool:
        """
        Add a stimulation device to the controller.
        
        Args:
            device_id: Unique identifier for the device.
            device: StimulatorInterface instance.
            
        Returns:
            bool: True if device was added successfully, False otherwise.
        """
        if device_id in self.devices:
            logger.warning(f"Device with ID {device_id} already exists.")
            return False
        
        self.devices[device_id] = {
            'interface': device,
            'connected': False,
            'battery_level': 0,
            'status': 'disconnected'
        }
        logger.info(f"Device {device_id} added to controller.")
        return True
    
    def remove_device(self, device_id: str) -> bool:
        """
        Remove a stimulation device from the controller.
        
        Args:
            device_id: Unique identifier for the device.
            
        Returns:
            bool: True if device was removed successfully, False otherwise.
        """
        if device_id not in self.devices:
            logger.warning(f"Device with ID {device_id} not found.")
            return False
        
        # Disconnect the device if it's connected
        if self.devices[device_id]['connected']:
            self.disconnect_device(device_id)
        
        del self.devices[device_id]
        logger.info(f"Device {device_id} removed from controller.")
        return True
    
    def connect_device(self, device_id: str) -> bool:
        """
        Connect to a stimulation device.
        
        Args:
            device_id: Unique identifier for the device.
            
        Returns:
            bool: True if device was connected successfully, False otherwise.
        """
        if device_id not in self.devices:
            logger.warning(f"Device with ID {device_id} not found.")
            return False
        
        device = self.devices[device_id]['interface']
        try:
            success = device.connect()
            if success:
                self.devices[device_id]['connected'] = True
                self.devices[device_id]['status'] = 'connected'
                # Update battery level
                self.devices[device_id]['battery_level'] = device.get_battery_level()
                logger.info(f"Device {device_id} connected successfully.")
            else:
                logger.error(f"Failed to connect to device {device_id}.")
            return success
        except Exception as e:
            logger.error(f"Error connecting to device {device_id}: {str(e)}")
            return False
    
    def disconnect_device(self, device_id: str) -> bool:
        """
        Disconnect from a stimulation device.
        
        Args:
            device_id: Unique identifier for the device.
            
        Returns:
            bool: True if device was disconnected successfully, False otherwise.
        """
        if device_id not in self.devices:
            logger.warning(f"Device with ID {device_id} not found.")
            return False
        
        if not self.devices[device_id]['connected']:
            logger.warning(f"Device {device_id} is not connected.")
            return True
        
        device = self.devices[device_id]['interface']
        try:
            success = device.disconnect()
            if success:
                self.devices[device_id]['connected'] = False
                self.devices[device_id]['status'] = 'disconnected'
                logger.info(f"Device {device_id} disconnected successfully.")
            else:
                logger.error(f"Failed to disconnect from device {device_id}.")
            return success
        except Exception as e:
            logger.error(f"Error disconnecting from device {device_id}: {str(e)}")
            return False
    
    def get_device_status(self, device_id: str) -> Dict[str, Union[str, float, bool]]:
        """
        Get the status of a stimulation device.
        
        Args:
            device_id: Unique identifier for the device.
            
        Returns:
            dict: Dictionary containing device status information.
        """
        if device_id not in self.devices:
            logger.warning(f"Device with ID {device_id} not found.")
            return {'error': 'Device not found'}
        
        device_info = self.devices[device_id].copy()
        device_info.pop('interface')  # Remove the interface object from the returned dict
        
        # If device is connected, update battery level
        if device_info['connected']:
            try:
                device_info['battery_level'] = self.devices[device_id]['interface'].get_battery_level()
            except Exception as e:
                logger.error(f"Error getting battery level for device {device_id}: {str(e)}")
        
        return device_info
    
    def get_all_devices_status(self) -> Dict[str, Dict[str, Union[str, float, bool]]]:
        """
        Get the status of all stimulation devices.
        
        Returns:
            dict: Dictionary with device IDs as keys and device status dictionaries as values.
        """
        statuses = {}
        for device_id in self.devices:
            statuses[device_id] = self.get_device_status(device_id)
        return statuses
    
    def synchronize_devices(self) -> bool:
        """
        Synchronize all connected devices.
        
        This implements the synchronization functionality described in 
        patent 10-2459338 ("저주파 자극기 제어장치").
        
        Returns:
            bool: True if all devices were synchronized successfully, False otherwise.
        """
        success = True
        for device_id, device_data in self.devices.items():
            if device_data['connected']:
                try:
                    device = device_data['interface']
                    if not device.synchronize():
                        logger.warning(f"Failed to synchronize device {device_id}")
                        success = False
                except Exception as e:
                    logger.error(f"Error synchronizing device {device_id}: {str(e)}")
                    success = False
        
        if success:
            logger.info("All devices synchronized successfully")
        return success
    
    def start_stimulation(self, 
                         anxiety_level: int, 
                         session_duration: int = 30, 
                         waveform: WaveformType = WaveformType.SINE, 
                         base_frequency: float = 0.5, 
                         phase_difference: float = 0.5, 
                         intensities: Optional[Dict[str, float]] = None) -> bool:
        """
        Start cranial electrotherapy stimulation session.
        
        This implements the stimulation functionality described in 
        patent 10-2022-0007209 ("불안장애 예방장치") with multiple devices
        and phase differences as in patent 10-2459338 ("저주파 자극기 제어장치").
        
        Args:
            anxiety_level: Detected anxiety level (0-3).
            session_duration: Duration of the stimulation session in minutes.
            waveform: Type of waveform to use.
            base_frequency: Base frequency of the stimulation in Hz.
            phase_difference: Phase difference between stimulation devices in seconds.
            intensities: Dictionary mapping device IDs to intensity levels (0.0-1.0).
                        If None, intensity will be determined based on anxiety level.
                        
        Returns:
            bool: True if stimulation started successfully, False otherwise.
        """
        # Check if already stimulating
        if self.is_stimulating:
            logger.warning("A stimulation session is already in progress")
            return False
        
        # Check if we have at least one connected device
        connected_devices = [d_id for d_id, d_data in self.devices.items() if d_data['connected']]
        if not connected_devices:
            logger.error("No connected devices available for stimulation")
            return False
        
        # Determine stimulation parameters based on anxiety level
        self.current_session = {
            'anxiety_level': anxiety_level,
            'start_time': time.time(),
            'duration': session_duration * 60,  # Convert to seconds
            'waveform': waveform,
            'base_frequency': base_frequency,
            'phase_difference': phase_difference,
            'devices': connected_devices
        }
        
        # Calculate appropriate intensity for each device
        if intensities is None:
            intensity_base = 0.25 + (anxiety_level * 0.15)  # Base intensity increases with anxiety level
            intensities = {}
            for i, device_id in enumerate(connected_devices):
                # Scale intensity slightly for each device
                intensities[device_id] = min(1.0, intensity_base - (i * 0.05))
        
        self.current_session['intensities'] = intensities
        
        # Set the stop event flag to False
        self.stop_event.clear()
        
        # Start the stimulation thread
        self.stimulation_thread = threading.Thread(
            target=self._stimulation_worker, 
            args=(self.current_session, self.stop_event)
        )
        self.stimulation_thread.daemon = True
        self.stimulation_thread.start()
        
        self.is_stimulating = True
        logger.info(f"Started stimulation session with anxiety level {anxiety_level}")
        return True
    
    def stop_stimulation(self) -> bool:
        """
        Stop the current stimulation session.
        
        Returns:
            bool: True if stimulation was stopped successfully, False otherwise.
        """
        if not self.is_stimulating:
            logger.warning("No stimulation session is currently in progress")
            return False
        
        # Set the stop event flag
        self.stop_event.set()
        
        # Wait for the stimulation thread to finish
        if self.stimulation_thread and self.stimulation_thread.is_alive():
            self.stimulation_thread.join(timeout=5.0)
        
        # Stop stimulation on all connected devices
        for device_id, device_data in self.devices.items():
            if device_data['connected']:
                try:
                    device_data['interface'].stop_stimulation()
                except Exception as e:
                    logger.error(f"Error stopping stimulation on device {device_id}: {str(e)}")
        
        self.is_stimulating = False
        self.current_session = None
        logger.info("Stopped stimulation session")
        return True
    
    def get_stimulation_status(self) -> Dict[str, Union[bool, Dict, None]]:
        """
        Get the status of the current stimulation session.
        
        Returns:
            dict: Dictionary containing stimulation status information.
        """
        status = {
            'is_stimulating': self.is_stimulating,
            'session': None
        }
        
        if self.is_stimulating and self.current_session:
            elapsed_time = time.time() - self.current_session['start_time']
            remaining_time = max(0, self.current_session['duration'] - elapsed_time)
            progress = min(100, (elapsed_time / self.current_session['duration']) * 100)
            
            status['session'] = {
                'anxiety_level': self.current_session['anxiety_level'],
                'elapsed_time': elapsed_time,
                'remaining_time': remaining_time,
                'progress': progress,
                'waveform': self.current_session['waveform'].value,
                'devices': self.current_session['devices'],
                'intensities': self.current_session['intensities']
            }
        
        return status
    
    def _stimulation_worker(self, session: Dict, stop_event: threading.Event):
        """
        Worker function for the stimulation thread.
        
        This function runs in a separate thread and controls the stimulation
        devices according to the session parameters.
        
        Args:
            session: Dictionary containing session parameters.
            stop_event: Event to signal thread termination.
        """
        try:
            # Configure stimulation parameters for each device
            devices = session['devices']
            for i, device_id in enumerate(devices):
                if device_id not in self.devices or not self.devices[device_id]['connected']:
                    continue
                
                device = self.devices[device_id]['interface']
                
                # Calculate phase offset for this device
                phase_offset = i * session['phase_difference']
                
                # Set stimulation parameters
                device.set_waveform(session['waveform'].value)
                device.set_frequency(session['base_frequency'])
                device.set_intensity(session['intensities'][device_id])
                device.set_phase_offset(phase_offset)
                
                # Start stimulation
                device.start_stimulation()
                logger.info(f"Started stimulation on device {device_id} with phase offset {phase_offset}")
            
            # Wait for the session to complete or stop event to be set
            start_time = time.time()
            while (time.time() - start_time) < session['duration'] and not stop_event.is_set():
                time.sleep(0.5)  # Check every 500ms
                
                # Adjust parameters based on elapsed time if needed
                elapsed_ratio = (time.time() - start_time) / session['duration']
                if elapsed_ratio > 0.75:
                    # Gradually reduce intensity in the last quarter of the session
                    for device_id in devices:
                        if device_id not in self.devices or not self.devices[device_id]['connected']:
                            continue
                        
                        device = self.devices[device_id]['interface']
                        original_intensity = session['intensities'][device_id]
                        fade_ratio = (1.0 - elapsed_ratio) / 0.25  # 1.0 to 0.0 in the last quarter
                        new_intensity = original_intensity * fade_ratio
                        device.set_intensity(new_intensity)
            
            # Stop stimulation on all devices
            for device_id in devices:
                if device_id not in self.devices or not self.devices[device_id]['connected']:
                    continue
                
                device = self.devices[device_id]['interface']
                device.stop_stimulation()
                logger.info(f"Stopped stimulation on device {device_id}")
                
        except Exception as e:
            logger.error(f"Error in stimulation worker: {str(e)}")
        finally:
            # Ensure we stop stimulation on all devices even if an error occurred
            for device_id in session['devices']:
                if device_id in self.devices and self.devices[device_id]['connected']:
                    try:
                        self.devices[device_id]['interface'].stop_stimulation()
                    except Exception as e:
                        logger.error(f"Error stopping stimulation on device {device_id}: {str(e)}")
            
            self.is_stimulating = False
            logger.info("Stimulation session complete")
            
    def get_recommended_parameters(self, anxiety_level: int) -> Dict[str, Union[str, float, int]]:
        """
        Get recommended stimulation parameters based on anxiety level.
        
        Args:
            anxiety_level: Detected anxiety level (0-3).
            
        Returns:
            dict: Dictionary containing recommended stimulation parameters.
        """
        # Define base parameters for each anxiety level
        if anxiety_level == 0:  # Low anxiety
            params = {
                'duration': 15,  # minutes
                'waveform': WaveformType.SINE.value,
                'frequency': 0.5,  # Hz
                'intensity_base': 0.3,
                'phase_difference': 0.3  # seconds
            }
        elif anxiety_level == 1:  # Moderate anxiety
            params = {
                'duration': 20,
                'waveform': WaveformType.SINE.value,
                'frequency': 0.7,
                'intensity_base': 0.4,
                'phase_difference': 0.4
            }
        elif anxiety_level == 2:  # High anxiety
            params = {
                'duration': 25,
                'waveform': WaveformType.BIPHASIC.value,
                'frequency': 0.9,
                'intensity_base': 0.5,
                'phase_difference': 0.5
            }
        else:  # Very high anxiety
            params = {
                'duration': 30,
                'waveform': WaveformType.BIPHASIC.value,
                'frequency': 1.0,
                'intensity_base': 0.6,
                'phase_difference': 0.7
            }
        
        return params
