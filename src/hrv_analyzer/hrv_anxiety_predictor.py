import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import neurokit2 as nk
import warnings
import logging

logger = logging.getLogger(__name__)

class HRVAnxietyPredictor:
    """
    심박변이도(HRV) 분석을 통한 불안장애 발생 가능성을 예측하는 클래스.
    
    특허 10-2022-0007209에 기반하여 구현되었으며,
    심전도에서 추출된 HRV 데이터를 분석하여 SampEn 등 다양한 지표를 계산하고
    불안장애 발생 가능성을 예측합니다.
    """
    
    def __init__(self, 
                 sampling_rate=256, 
                 control_sampen_mean=1.4, 
                 anxiety_threshold=0.75,
                 window_size=300,
                 hrv_params=None):
        """
        HRVAnxietyPredictor 초기화 함수
        
        매개변수:
            sampling_rate (int): ECG 신호의 샘플링 레이트 (Hz)
            control_sampen_mean (float): 대조군의 SampEn 평균값
            anxiety_threshold (float): 불안장애 임계값 (0-1 사이)
            window_size (int): 분석 윈도우 크기 (초)
            hrv_params (dict): HRV 분석 매개변수
        """
        self.sampling_rate = sampling_rate
        self.control_sampen_mean = control_sampen_mean
        self.anxiety_threshold = anxiety_threshold
        self.window_size = window_size
        
        # 기본 HRV 분석 매개변수
        self.hrv_params = {
            'sampen_m': 2,          # 패턴 길이
            'sampen_r': 0.2,         # 유사성 임계값
            'lf_band': (0.04, 0.15), # 저주파 대역 (Hz)
            'hf_band': (0.15, 0.4),  # 고주파 대역 (Hz)
            'vlf_band': (0.0033, 0.04), # 초저주파 대역 (Hz)
        }
        
        # 사용자 정의 매개변수로 업데이트
        if hrv_params:
            self.hrv_params.update(hrv_params)
        
        self.history = {
            'timestamp': [],
            'rmssd': [],
            'sdnn': [],
            'lf_hf_ratio': [],
            'sampen': [],
            'anxiety_score': [],
        }
        
        logger.info("HRVAnxietyPredictor 초기화 완료")
    
    def extract_hrv_features(self, rri):
        """
        R-R 간격에서 HRV 특성을 추출합니다.
        
        매개변수:
            rri (array): R-R 간격 시리즈 (ms)
            
        반환값:
            dict: HRV 특성
        """
        if len(rri) < 10:
            logger.warning(f"RR 간격이 너무 적습니다: {len(rri)}")
            return None
        
        try:
            # 시간 영역 분석
            rmssd = np.sqrt(np.mean(np.diff(rri) ** 2))  # RMSSD
            sdnn = np.std(rri)  # SDNN
            
            # 주파수 영역 분석
            try:
                # R-R 시리즈를 5Hz로 리샘플링
                rri_interp = nk.signal_interpolate(
                    rri, 
                    x_new=np.arange(0, len(rri) / 5, 1/5), 
                    method='cubic'
                )
                
                # 주파수 영역 파워 스펙트럼 밀도 계산
                freq, psd = signal.welch(
                    rri_interp, 
                    fs=5.0, 
                    nperseg=len(rri_interp) // 2,
                    scaling='density'
                )
                
                # 각 주파수 대역 파워 계산
                vlf_mask = (freq >= self.hrv_params['vlf_band'][0]) & (freq < self.hrv_params['vlf_band'][1])
                lf_mask = (freq >= self.hrv_params['lf_band'][0]) & (freq < self.hrv_params['lf_band'][1])
                hf_mask = (freq >= self.hrv_params['hf_band'][0]) & (freq < self.hrv_params['hf_band'][1])
                
                vlf_power = np.trapz(psd[vlf_mask], freq[vlf_mask])
                lf_power = np.trapz(psd[lf_mask], freq[lf_mask])
                hf_power = np.trapz(psd[hf_mask], freq[hf_mask])
                
                # LF/HF 비율 계산
                lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
                
                # 총 파워 계산
                total_power = vlf_power + lf_power + hf_power
                
                # 정규화된 파워 계산
                norm_lf = 100 * lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
                norm_hf = 100 * hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
                
            except Exception as e:
                logger.warning(f"주파수 영역 분석 중 오류 발생: {e}")
                lf_power, hf_power, norm_lf, norm_hf, lf_hf_ratio, total_power = 0, 0, 0, 0, 0, 0
            
            # 비선형 분석 - SampEn 계산
            try:
                sampen = self._calculate_sample_entropy(rri)
            except Exception as e:
                logger.warning(f"SampEn 계산 중 오류 발생: {e}")
                sampen = 0
            
            features = {
                'rmssd': rmssd,
                'sdnn': sdnn,
                'lf_power': lf_power,
                'hf_power': hf_power,
                'lf_hf_ratio': lf_hf_ratio,
                'norm_lf': norm_lf,
                'norm_hf': norm_hf,
                'total_power': total_power,
                'sampen': sampen
            }
            
            return features
            
        except Exception as e:
            logger.error(f"HRV 특성 추출 중 오류 발생: {e}")
            return None
    
    def _calculate_sample_entropy(self, rri):
        """
        R-R 간격의 Sample Entropy(SampEn)를 계산합니다.
        
        매개변수:
            rri (array): R-R 간격 시리즈 (ms)
            
        반환값:
            float: Sample Entropy 값
        """
        m = self.hrv_params['sampen_m']
        r = self.hrv_params['sampen_r'] * np.std(rri)
        
        # SampEn 계산 (nk2 라이브러리 사용)
        try:
            sampen = nk.entropy_sample(rri, dimension=m, tolerance=r)
        except Exception as e:
            # 직접 계산 시도
            logger.warning(f"NK2로 SampEn 계산 실패, 직접 계산 시도: {e}")
            N = len(rri)
            
            # m 및 m+1 길이 벡터 생성
            def _create_templates(data, m):
                templates = []
                for i in range(len(data) - m + 1):
                    templates.append(data[i:i+m])
                return np.array(templates)
            
            # 두 벡터 사이의 Chebyshev 거리
            def _chebyshev_distance(x, y):
                return np.max(np.abs(x - y))
            
            # m 및 m+1 길이 벡터 유사성 개수 계산
            templates_m = _create_templates(rri, m)
            templates_m1 = _create_templates(rri, m+1)
            
            # B_m(r) 계산
            B = 0
            for i in range(N - m):
                # 자기 자신 제외
                matches = 0
                for j in range(N - m):
                    if i == j:
                        continue
                    if _chebyshev_distance(templates_m[i], templates_m[j]) <= r:
                        matches += 1
                B += matches
            B = B / ((N - m) * (N - m - 1))
            
            # A_m(r) 계산
            A = 0
            for i in range(N - m - 1):
                matches = 0
                for j in range(N - m - 1):
                    if i == j:
                        continue
                    if _chebyshev_distance(templates_m1[i], templates_m1[j]) <= r:
                        matches += 1
                A += matches
            A = A / ((N - m - 1) * (N - m - 2))
            
            # SampEn = -log(A/B)
            sampen = -np.log(A / B) if B > 0 and A > 0 else 0
        
        return sampen
    
    def predict_anxiety(self, rri, timestamp=None):
        """
        R-R 간격 데이터를 분석하여 불안장애 발생 가능성을 예측합니다.
        
        매개변수:
            rri (array): R-R 간격 시리즈 (ms)
            timestamp (float, optional): 타임스탬프
            
        반환값:
            dict: 불안 예측 결과
        """
        # HRV 특성 추출
        features = self.extract_hrv_features(rri)
        if features is None:
            return {
                'anxiety_score': 0,
                'prediction': 'insufficient_data',
                'features': {},
                'stimulation_recommended': False
            }
        
        # 불안 점수 계산 (특허 10-2022-0007209 기반)
        # SampEn 값이 대조군 평균보다 작을수록 불안장애 가능성이 높음
        sampen_ratio = features['sampen'] / self.control_sampen_mean
        
        # RMSSD 및 SDNN 저하도 불안 증상 징후
        rmssd_weight = min(1, features['rmssd'] / 50.0)  # 정상 성인 RMSSD 약 50ms
        sdnn_weight = min(1, features['sdnn'] / 100.0)   # 정상 성인 SDNN 약 100ms
        
        # LF/HF 비율 증가시 교감신경 활성화, 불안 증가 가능성
        lf_hf_weight = 2.0 / (1.0 + min(features['lf_hf_ratio'], 4.0))
        
        # 불안 점수 계산 (0-1 범위, 1에 가까울수록 불안 수준 높음)
        anxiety_score = 1.0 - (0.5 * sampen_ratio + 0.25 * rmssd_weight + 0.25 * sdnn_weight)
        anxiety_score = max(0, min(1, anxiety_score))  # 0-1 범위로 제한
        
        # 불안 수준 분류
        if anxiety_score < 0.3:
            prediction = 'normal'
            stimulation_recommended = False
        elif anxiety_score < self.anxiety_threshold:
            prediction = 'mild_anxiety'
            stimulation_recommended = True
        else:
            prediction = 'high_anxiety'
            stimulation_recommended = True
        
        # 이력 기록
        if timestamp is not None:
            self.history['timestamp'].append(timestamp)
            self.history['rmssd'].append(features['rmssd'])
            self.history['sdnn'].append(features['sdnn'])
            self.history['lf_hf_ratio'].append(features['lf_hf_ratio'])
            self.history['sampen'].append(features['sampen'])
            self.history['anxiety_score'].append(anxiety_score)
        
        # 결과 반환
        result = {
            'anxiety_score': anxiety_score,
            'prediction': prediction,
            'features': features,
            'stimulation_recommended': stimulation_recommended
        }
        
        return result
    
    def get_history(self):
        """
        분석 이력을 반환합니다.
        
        반환값:
            pandas.DataFrame: 분석 이력
        """
        return pd.DataFrame(self.history)
    
    def export_history(self, filename):
        """
        분석 이력을 CSV 파일로 내보냅니다.
        
        매개변수:
            filename (str): 파일 이름
        """
        df = self.get_history()
        df.to_csv(filename, index=False)
        logger.info(f"분석 이력이 {filename}에 저장되었습니다.")
    
    def clear_history(self):
        """
        분석 이력을 초기화합니다.
        """
        for key in self.history:
            self.history[key] = []
        logger.info("분석 이력이 초기화되었습니다.")