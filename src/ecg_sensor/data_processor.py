"""
ECG Data Processor

This module processes raw ECG data from sensors, handling filtering,
R-peak detection, and preparing data for HRV analysis.
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging
import json
import time
import os
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECGDataProcessor:
    """
    Processes raw ECG data for various analyses, including:
    - Signal filtering and normalization
    - R-peak detection
    - Artifact removal
    - Data preparation for HRV analysis
    """
    
    def __init__(self, sample_rate: int = 250):
        """
        Initialize the ECG data processor.
        
        Args:
            sample_rate: Sampling rate of the ECG data in Hz
        """
        self.sample_rate = sample_rate
        
        # Buffer for raw ECG data
        self.raw_buffer = []
        self.processed_buffer = []
        
        # Buffer size (10 seconds of data)
        self.buffer_size = 10 * sample_rate
        
        # Callbacks
        self.data_processed_callback = None
        self.r_peak_detected_callback = None
        
        # Filters
        self._init_filters()
        
        # R-peak detection parameters
        self.r_peak_threshold = 0.6  # Adaptive threshold
        self.last_r_peaks = []
        self.min_r_peak_distance = int(0.25 * sample_rate)  # Minimum distance between R-peaks (250ms)
    
    def _init_filters(self):
        """Initialize signal processing filters."""
        # Bandpass filter for removing baseline wander and high-frequency noise
        # Typical ECG frequency range is 0.5-40 Hz
        nyquist = 0.5 * self.sample_rate
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        # Design bandpass filter
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
        # Notch filter for removing power line interference (50/60 Hz)
        # We'll use 50 Hz for EU/Asia and 60 Hz for North America
        for freq in [50, 60]:
            f0 = freq / nyquist
            q = 30.0  # Quality factor
            self.notch_b, self.notch_a = signal.iirnotch(f0, q)
    
    def process_raw_data(self, raw_data: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Process raw ECG data through filtering pipeline.
        
        Args:
            raw_data: Raw ECG signal
            
        Returns:
            Processed ECG signal
        """
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
        
        # Add to buffer
        self.raw_buffer.extend(raw_data)
        
        # Keep buffer at fixed size
        if len(self.raw_buffer) > self.buffer_size:
            self.raw_buffer = self.raw_buffer[-self.buffer_size:]
        
        # Convert buffer to numpy array
        signal_data = np.array(self.raw_buffer)
        
        # Apply filters
        try:
            # Remove baseline wander and high-frequency noise with bandpass filter
            filtered_data = signal.filtfilt(self.b, self.a, signal_data)
            
            # Remove power line interference with notch filters
            filtered_data = signal.filtfilt(self.notch_b, self.notch_a, filtered_data)
            
            # Normalize
            filtered_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
            
            # Store in processed buffer
            self.processed_buffer = filtered_data.tolist()
            
            # Call callback if registered
            if self.data_processed_callback:
                self.data_processed_callback(filtered_data)
            
            # Return the processed data
            return filtered_data
        
        except Exception as e:
            logger.error(f"Error processing ECG data: {str(e)}")
            return np.array(signal_data)  # Return original data in case of error
    
    def detect_r_peaks(self, processed_data: Optional[np.ndarray] = None) -> List[int]:
        """
        Detect R-peaks in the processed ECG signal.
        
        Args:
            processed_data: Processed ECG signal (if None, use the internal buffer)
            
        Returns:
            List of R-peak indices
        """
        if processed_data is None:
            if not self.processed_buffer:
                logger.warning("No processed data available for R-peak detection")
                return []
            processed_data = np.array(self.processed_buffer)
        
        try:
            # Use Pan-Tompkins algorithm for R-peak detection
            # 1. Apply derivative filter to emphasize QRS complex
            derivative = np.diff(processed_data)
            squared = derivative ** 2
            
            # 2. Moving window integration
            window_size = int(0.15 * self.sample_rate)  # 150ms window
            integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
            
            # 3. Adaptive thresholding
            # Calculate mean of the integrated signal for adaptive threshold
            signal_mean = np.mean(integrated)
            threshold = signal_mean * self.r_peak_threshold
            
            # 4. Find peaks above threshold
            peaks, _ = signal.find_peaks(
                integrated, 
                height=threshold,
                distance=self.min_r_peak_distance
            )
            
            # 5. Adjust peak positions to match actual R-peaks
            # Often, the integrated signal peaks are slightly offset from actual R-peaks
            r_peaks = []
            for peak in peaks:
                # Look for the maximum value in a small window around each peak
                start = max(0, peak - int(0.025 * self.sample_rate))
                end = min(len(processed_data) - 1, peak + int(0.025 * self.sample_rate))
                
                if start < end:
                    window = processed_data[start:end]
                    max_idx = np.argmax(window)
                    r_peaks.append(start + max_idx)
            
            # Store detected R-peaks
            self.last_r_peaks = r_peaks
            
            # Call callback if registered
            if self.r_peak_detected_callback:
                self.r_peak_detected_callback(r_peaks)
            
            return r_peaks
        
        except Exception as e:
            logger.error(f"Error detecting R-peaks: {str(e)}")
            return []
    
    def calculate_rr_intervals(self, r_peaks: Optional[List[int]] = None) -> List[float]:
        """
        Calculate RR intervals from detected R-peaks.
        
        Args:
            r_peaks: List of R-peak indices (if None, use last detected peaks)
            
        Returns:
            List of RR intervals in milliseconds
        """
        if r_peaks is None:
            r_peaks = self.last_r_peaks
        
        if len(r_peaks) < 2:
            logger.warning("Not enough R-peaks to calculate RR intervals")
            return []
        
        # Calculate intervals between consecutive R-peaks
        rr_intervals = []
        for i in range(1, len(r_peaks)):
            # Convert sample difference to milliseconds
            rr_ms = (r_peaks[i] - r_peaks[i-1]) * (1000 / self.sample_rate)
            rr_intervals.append(rr_ms)
        
        return rr_intervals
    
    def filter_rr_intervals(self, rr_intervals: List[float]) -> List[float]:
        """
        Filter RR intervals to remove artifacts.
        
        Args:
            rr_intervals: List of RR intervals in milliseconds
            
        Returns:
            Filtered RR intervals
        """
        if not rr_intervals:
            return []
        
        # Calculate statistics
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        
        # Define acceptable range (typically mean ± 20% or more)
        # For more severe filtering, use smaller percentages
        lower_limit = rr_mean - 0.2 * rr_mean
        upper_limit = rr_mean + 0.2 * rr_mean
        
        # Filter outliers
        filtered_rr = [rr for rr in rr_intervals if lower_limit <= rr <= upper_limit]
        
        if len(filtered_rr) < len(rr_intervals):
            logger.info(f"Filtered out {len(rr_intervals) - len(filtered_rr)} RR intervals")
        
        return filtered_rr
    
    def prepare_for_hrv_analysis(self, raw_data: Union[List[float], np.ndarray]) -> Dict:
        """
        Prepare ECG data for HRV analysis by processing it through the entire pipeline.
        
        Args:
            raw_data: Raw ECG signal
            
        Returns:
            Dictionary containing processed data, R-peaks, and RR intervals
        """
        # Process the data
        processed_data = self.process_raw_data(raw_data)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(processed_data)
        
        # Calculate RR intervals
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        
        # Filter RR intervals
        filtered_rr = self.filter_rr_intervals(rr_intervals)
        
        return {
            'processed_data': processed_data.tolist(),
            'r_peaks': r_peaks,
            'rr_intervals': rr_intervals,
            'filtered_rr_intervals': filtered_rr,
            'sample_rate': self.sample_rate,
            'timestamp': time.time()
        }
    
    def set_data_processed_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for when data is processed."""
        self.data_processed_callback = callback
    
    def set_r_peak_detected_callback(self, callback: Callable[[List[int]], None]):
        """Set callback for when R-peaks are detected."""
        self.r_peak_detected_callback = callback
    
    def save_data_to_file(self, data: Dict, filename: str = None):
        """
        Save ECG data to a file.
        
        Args:
            data: Dictionary containing ECG data
            filename: Output filename (if None, generate automatically)
        """
        if filename is None:
            # Generate filename based on current time
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ecg_data_{timestamp}.json"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Saved ECG data to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving ECG data to file: {str(e)}")
    
    def load_data_from_file(self, filename: str) -> Dict:
        """
        Load ECG data from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary containing ECG data
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded ECG data from {filename}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading ECG data from file: {str(e)}")
            return {}


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate synthetic ECG data for testing
    def generate_synthetic_ecg(duration=10, rate=250):
        """Generate synthetic ECG data."""
        t = np.arange(0, duration, 1.0/rate)
        # Base heart rate
        hr = 60  # bpm
        # Convert to frequency
        freq = hr / 60.0
        # Basic sine wave with heart frequency
        ecg = np.sin(2 * np.pi * freq * t)
        
        # Add QRS complex
        for i in range(int(duration * freq)):
            center = i / freq
            # R peak
            r_peak = np.exp(-((t - center) ** 2) / (2 * 0.01 ** 2))
            ecg += r_peak * 2
            
            # Q and S waves
            q_center = center - 0.025
            s_center = center + 0.025
            q_wave = -np.exp(-((t - q_center) ** 2) / (2 * 0.015 ** 2)) * 0.5
            s_wave = -np.exp(-((t - s_center) ** 2) / (2 * 0.015 ** 2)) * 0.5
            ecg += q_wave + s_wave
            
            # T wave
            t_center = center + 0.15
            t_wave = np.exp(-((t - t_center) ** 2) / (2 * 0.07 ** 2)) * 0.75
            ecg += t_wave
        
        # Add noise
        noise = np.random.normal(0, 0.05, len(t))
        ecg += noise
        
        # Add baseline wander
        baseline = 0.5 * np.sin(2 * np.pi * 0.05 * t)
        ecg += baseline
        
        return ecg
    
    # Create processor
    processor = ECGDataProcessor(sample_rate=250)
    
    # Generate synthetic data
    raw_ecg = generate_synthetic_ecg(duration=10, rate=250)
    
    # Process the data
    result = processor.prepare_for_hrv_analysis(raw_ecg)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot raw and processed ECG
    plt.subplot(2, 1, 1)
    plt.plot(raw_ecg, label='Raw ECG')
    plt.plot(result['processed_data'], label='Processed ECG')
    plt.scatter([result['r_peaks']], [np.array(result['processed_data'])[result['r_peaks']]], 
                color='red', marker='x', label='Detected R-peaks')
    plt.legend()
    plt.title('ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot RR intervals
    plt.subplot(2, 1, 2)
    plt.plot(result['rr_intervals'], label='RR Intervals')
    plt.plot(result['filtered_rr_intervals'], label='Filtered RR Intervals')
    plt.axhline(y=np.mean(result['rr_intervals']), color='r', linestyle='--', 
                label=f'Mean RR: {np.mean(result["rr_intervals"]):.2f} ms')
    plt.legend()
    plt.title('RR Intervals')
    plt.xlabel('Beat Number')
    plt.ylabel('RR Interval (ms)')
    
    plt.tight_layout()
    plt.show()
    
    # Print heart rate statistics
    if result['filtered_rr_intervals']:
        mean_rr = np.mean(result['filtered_rr_intervals'])
        hr = 60000 / mean_rr  # Convert to bpm
        print(f"Mean RR Interval: {mean_rr:.2f} ms")
        print(f"Heart Rate: {hr:.2f} bpm")
