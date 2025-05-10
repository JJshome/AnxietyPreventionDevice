import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import neurokit2 as nk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HRVAnalyzer')

class HRVAnalyzer:
    """
    HRV(Heart Rate Variability) 분석을 위한 클래스
    
    특허 10-2022-0007209에 기반한 심전도 신호로부터 HRV 특성을 추출하여
    불안장애 예측에 활용할 수 있는 분석 도구를 제공합니다.
    """
    
    def __init__(self, sampling_rate=256, window_size=300, overlap=0.5):
        """
        HRVAnalyzer 초기화
        
        Args:
            sampling_rate (int): ECG 신호의 샘플링 레이트 (Hz)
            window_size (int): 분석 윈도우 크기 (초)
            overlap (float): 윈도우 중첩 비율 (0-1)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))
        
        logger.info(f"HRVAnalyzer initialized with sampling_rate={sampling_rate}Hz, "
                   f"window_size={window_size}s, overlap={overlap}")
    
    def preprocess_ecg(self, ecg_signal):
        """
        ECG 신호 전처리
        
        Args:
            ecg_signal (np.ndarray): 원시 ECG 신호
            
        Returns:
            np.ndarray: 전처리된 ECG 신호
        """
        # 신호 정규화
        ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # 대역 통과 필터 적용 (0.5-45Hz)
        b, a = signal.butter(3, [0.5, 45], fs=self.sampling_rate, btype='bandpass')
        ecg_filtered = signal.filtfilt(b, a, ecg_normalized)
        
        # 50Hz 노치 필터 (전원 노이즈 제거)
        b_notch, a_notch = signal.iirnotch(50, 30, self.sampling_rate)
        ecg_filtered = signal.filtfilt(b_notch, a_notch, ecg_filtered)
        
        logger.debug(f"ECG signal preprocessed, shape: {ecg_filtered.shape}")
        return ecg_filtered
    
    def detect_r_peaks(self, ecg_signal):
        """
        ECG 신호에서 R-peak 검출
        
        Args:
            ecg_signal (np.ndarray): 전처리된 ECG 신호
            
        Returns:
            np.ndarray: R-peak 위치 (샘플 인덱스)
        """
        # NeuroKit2 라이브러리 활용 R-peaks 검출
        _, info = nk.ecg_process(ecg_signal, sampling_rate=self.sampling_rate)
        r_peaks = info['ECG_R_Peaks']
        
        # 너무 가까운 피크 제거 (잘못된 검출 방지)
        if len(r_peaks) > 2:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # ms 단위로 변환
            valid_rr = (rr_intervals > 300) & (rr_intervals < 2000)  # 300ms-2000ms 범위만 유효
            r_peaks = r_peaks[:-1][valid_rr]  # 마지막 피크는 제외하고 유효한 RR 간격에 해당하는 피크만 선택
        
        logger.debug(f"Detected {len(r_peaks)} R-peaks")
        return r_peaks
    
    def calculate_rr_intervals(self, r_peaks):
        """
        R-peak 위치로부터 RR 간격 계산
        
        Args:
            r_peaks (np.ndarray): R-peak 위치 (샘플 인덱스)
            
        Returns:
            np.ndarray: RR 간격 (ms)
            np.ndarray: RR 간격 시간 위치 (s)
        """
        # 샘플 간격을 시간 간격으로 변환 (ms)
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        # RR 간격의 시간 위치 (초)
        rr_times = r_peaks[1:] / self.sampling_rate
        
        return rr_intervals, rr_times
    
    def interpolate_rr_intervals(self, rr_intervals, rr_times, fs=4.0):
        """
        RR 간격 보간
        
        Args:
            rr_intervals (np.ndarray): RR 간격 (ms)
            rr_times (np.ndarray): RR 간격 시간 위치 (s)
            fs (float): 원하는 샘플링 주파수 (Hz)
            
        Returns:
            np.ndarray: 보간된 RR 간격
            np.ndarray: 보간된 시간 위치
        """
        if len(rr_intervals) < 2:
            logger.warning("Not enough RR intervals for interpolation")
            return np.array([]), np.array([])
        
        # 보간 함수 생성
        interpolation_func = interp1d(rr_times, rr_intervals, kind='cubic', 
                                      bounds_error=False, fill_value="extrapolate")
        
        # 균일한 시간 간격 생성
        t_interpol = np.arange(rr_times[0], rr_times[-1], 1/fs)
        
        # 보간 적용
        rr_interpol = interpolation_func(t_interpol)
        
        return rr_interpol, t_interpol
    
    def calculate_time_domain_features(self, rr_intervals):
        """
        시간 도메인 HRV 특성 계산
        
        Args:
            rr_intervals (np.ndarray): RR 간격 (ms)
            
        Returns:
            dict: 시간 도메인 HRV 특성
        """
        if len(rr_intervals) < 2:
            logger.warning("Not enough RR intervals for time-domain analysis")
            return {}
        
        features = {}
        
        # 기본 통계 특성
        features['mean_rr'] = np.mean(rr_intervals)  # 평균 RR 간격 (ms)
        features['sdnn'] = np.std(rr_intervals)      # SDNN (ms)
        features['hr'] = 60000 / features['mean_rr']  # 평균 심박수 (bpm)
        
        # RMSSD (연속 RR 간격 차이 제곱근의 평균)
        rr_diff = np.diff(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(rr_diff**2))
        
        # pNN50 (50ms 이상 차이나는 연속 RR 간격의 비율)
        nn50 = np.sum(np.abs(rr_diff) > 50)
        features['pnn50'] = 100 * nn50 / len(rr_diff) if len(rr_diff) > 0 else 0
        
        # 변동 계수
        features['cov'] = features['sdnn'] / features['mean_rr'] * 100
        
        return features
    
    def calculate_frequency_domain_features(self, rr_interpol, t_interpol, fs=4.0):
        """
        주파수 도메인 HRV 특성 계산
        
        Args:
            rr_interpol (np.ndarray): 보간된 RR 간격
            t_interpol (np.ndarray): 보간된 시간 위치
            fs (float): 보간된 신호의 샘플링 주파수
            
        Returns:
            dict: 주파수 도메인 HRV 특성
        """
        if len(rr_interpol) < fs * 10:  # 최소 10초 데이터 필요
            logger.warning("Not enough data for frequency-domain analysis")
            return {}
        
        features = {}
        
        # 추세 제거
        rr_detrended = signal.detrend(rr_interpol)
        
        # 파워 스펙트럼 밀도 계산
        f, psd = signal.welch(rr_detrended, fs=fs, nperseg=len(rr_detrended)//2, scaling='density')
        
        # 주파수 대역 인덱스
        vlf_idx = np.logical_and(f >= 0.0033, f < 0.04)  # Very Low Frequency (0.0033-0.04Hz)
        lf_idx = np.logical_and(f >= 0.04, f < 0.15)     # Low Frequency (0.04-0.15Hz)
        hf_idx = np.logical_and(f >= 0.15, f < 0.4)      # High Frequency (0.15-0.4Hz)
        
        # 주파수 대역별 파워
        features['vlf_power'] = np.trapz(psd[vlf_idx], f[vlf_idx])
        features['lf_power'] = np.trapz(psd[lf_idx], f[lf_idx])
        features['hf_power'] = np.trapz(psd[hf_idx], f[hf_idx])
        features['total_power'] = features['vlf_power'] + features['lf_power'] + features['hf_power']
        
        # 정규화된 파워
        lf_hf_sum = features['lf_power'] + features['hf_power']
        if lf_hf_sum > 0:
            features['lf_nu'] = 100 * features['lf_power'] / lf_hf_sum
            features['hf_nu'] = 100 * features['hf_power'] / lf_hf_sum
            features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
        else:
            features['lf_nu'] = 0
            features['hf_nu'] = 0
            features['lf_hf_ratio'] = 0
        
        return features
    
    def calculate_nonlinear_features(self, rr_intervals):
        """
        비선형 HRV 특성 계산
        
        Args:
            rr_intervals (np.ndarray): RR 간격 (ms)
            
        Returns:
            dict: 비선형 HRV 특성
        """
        if len(rr_intervals) < 10:  # 최소 10개 RR 간격 필요
            logger.warning("Not enough RR intervals for nonlinear analysis")
            return {}
        
        features = {}
        
        # 비선형 특성 계산을 위해 NeuroKit2 사용
        hrv_nonlinear = nk.hrv_nonlinear(rr_intervals, sampling_rate=self.sampling_rate)
        
        # SD1, SD2 (푸앵카레 플롯 특성)
        features['sd1'] = hrv_nonlinear['HRV_SD1'].values[0] if len(hrv_nonlinear) > 0 else 0
        features['sd2'] = hrv_nonlinear['HRV_SD2'].values[0] if len(hrv_nonlinear) > 0 else 0
        features['sd_ratio'] = features['sd1'] / features['sd2'] if features['sd2'] > 0 else 0
        
        # 근사 엔트로피 (ApEn)와 샘플 엔트로피 (SampEn)
        # 참고: 특허 10-2016-0035318에 따라 SampEn이 불안장애 예측에 유용
        features['sampen'] = hrv_nonlinear['HRV_SampEn'].values[0] if len(hrv_nonlinear) > 0 else 0
        
        return features
    
    def calculate_hrv_features(self, ecg_signal):
        """
        ECG 신호에서 HRV 특성 계산
        
        Args:
            ecg_signal (np.ndarray): ECG 신호
            
        Returns:
            dict: 모든 HRV 특성
        """
        logger.info("Calculating HRV features...")
        
        # ECG 신호 전처리
        ecg_filtered = self.preprocess_ecg(ecg_signal)
        
        # R-peak 검출
        r_peaks = self.detect_r_peaks(ecg_filtered)
        
        if len(r_peaks) < 2:
            logger.warning("Not enough R-peaks detected for HRV analysis")
            return {}
        
        # RR 간격 계산
        rr_intervals, rr_times = self.calculate_rr_intervals(r_peaks)
        
        # 이상치 제거 (범위: 300-2000ms)
        valid_idx = np.logical_and(rr_intervals >= 300, rr_intervals <= 2000)
        rr_intervals_cleaned = rr_intervals[valid_idx]
        rr_times_cleaned = rr_times[valid_idx]
        
        if len(rr_intervals_cleaned) < 2:
            logger.warning("Not enough valid RR intervals for HRV analysis")
            return {}
        
        # RR 간격 보간
        rr_interpol, t_interpol = self.interpolate_rr_intervals(rr_intervals_cleaned, rr_times_cleaned)
        
        # 모든 특성 계산 및 통합
        features = {}
        
        # 시간 도메인 특성
        time_features = self.calculate_time_domain_features(rr_intervals_cleaned)
        features.update(time_features)
        
        # 주파수 도메인 특성
        freq_features = self.calculate_frequency_domain_features(rr_interpol, t_interpol)
        features.update(freq_features)
        
        # 비선형 특성
        nonlinear_features = self.calculate_nonlinear_features(rr_intervals_cleaned)
        features.update(nonlinear_features)
        
        # 추가 메타데이터
        features['n_beats'] = len(r_peaks)
        features['recording_length'] = len(ecg_signal) / self.sampling_rate
        features['analysis_time'] = rr_times_cleaned[-1] - rr_times_cleaned[0] if len(rr_times_cleaned) > 1 else 0
        
        logger.info(f"HRV features calculated: {len(features)} features")
        return features
    
    def analyze_windowed(self, ecg_signal):
        """
        ECG 신호를 윈도우로 나누어 HRV 특성 계산
        
        Args:
            ecg_signal (np.ndarray): ECG 신호
            
        Returns:
            list: 윈도우별 HRV 특성
        """
        # 신호 길이 확인
        signal_length = len(ecg_signal)
        
        if signal_length < self.window_samples:
            logger.warning(f"Signal too short ({signal_length} samples) for window analysis "
                          f"({self.window_samples} samples required)")
            # 전체 신호에 대해 분석
            return [self.calculate_hrv_features(ecg_signal)]
        
        # 윈도우 위치 계산
        window_starts = np.arange(0, signal_length - self.window_samples + 1, self.step_samples)
        
        logger.info(f"Analyzing signal with {len(window_starts)} windows")
        
        # 각 윈도우에 대해 분석
        window_features = []
        for i, start in enumerate(window_starts):
            end = start + self.window_samples
            window = ecg_signal[start:end]
            
            logger.debug(f"Analyzing window {i+1}/{len(window_starts)}: samples {start}-{end}")
            
            features = self.calculate_hrv_features(window)
            if features:  # 유효한 특성이 있는 경우에만 추가
                features['window_start'] = start / self.sampling_rate  # 윈도우 시작 시간 (초)
                features['window_end'] = end / self.sampling_rate  # 윈도우 종료 시간 (초)
                window_features.append(features)
        
        return window_features
    
    def plot_hrv_features(self, features_list, feature_names=None):
        """
        HRV 특성의 시간적 변화를 시각화
        
        Args:
            features_list (list): 윈도우별 HRV 특성 목록
            feature_names (list): 시각화할 특성 이름 목록 (기본값: 주요 특성)
        """
        if not features_list:
            logger.warning("No features to plot")
            return
        
        # 기본 특성 목록
        if feature_names is None:
            feature_names = ['hr', 'sdnn', 'rmssd', 'pnn50', 'lf_hf_ratio', 'sampen']
        
        # 데이터프레임 생성
        df = pd.DataFrame(features_list)
        
        # 시간 축 생성
        if 'window_start' in df.columns:
            time_axis = df['window_start'].values
        else:
            time_axis = np.arange(len(df))
        
        # 그래프 생성
        fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 3*len(feature_names)), sharex=True)
        
        for i, feature in enumerate(feature_names):
            if feature in df.columns:
                ax = axes[i] if len(feature_names) > 1 else axes
                ax.plot(time_axis, df[feature].values, 'o-', label=feature)
                ax.set_ylabel(feature)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
        
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        
        return fig