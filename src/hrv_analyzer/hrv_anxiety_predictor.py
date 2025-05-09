"""
HRV 기반 불안장애 예측 모듈 (특허 10-2022-0007209 기반)

심박변이도(HRV) 분석을 통해 불안장애 발생 가능성을 예측하는 기능을 구현합니다.
특허에 따라 SampEn(Sample Entropy)과 다양한 HRV 지표를 활용합니다.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
import joblib
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)


class AnxietyLevel(Enum):
    """불안 수준 분류"""
    NORMAL = 0      # 정상
    MILD = 1        # 경미한 불안
    MODERATE = 2    # 중등도 불안
    SEVERE = 3      # 심각한 불안
    PANIC = 4       # 패닉 수준


class HRVAnxietyPredictor:
    """
    심박변이도(HRV) 기반 불안장애 예측 클래스
    
    특허 10-2022-0007209에 기반한 구현으로, HRV 지표를 분석하여
    불안장애 발생 가능성을 예측합니다.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        불안장애 예측기 초기화
        
        Args:
            model_path: 사전 학습된 모델 경로 (None인 경우 기본 규칙 기반 모델 사용)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.use_rule_based = True
        
        # 모델 로드 시도
        if model_path:
            try:
                self.model = joblib.load(model_path)
                self.use_rule_based = False
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                logger.info("Falling back to rule-based prediction")
        
        # 대조군 HRV 기준값 (기준 논문 및 특허 내용 기반)
        self.reference_values = {
            "SampEn": 1.55,    # 정상 성인 대조군 SampEn 평균
            "SDNN": 65.0,      # 정상 성인 SDNN 평균 (ms)
            "RMSSD": 42.0,     # 정상 성인 RMSSD 평균 (ms)
            "pNN50": 23.0,     # 정상 성인 pNN50 평균 (%)
            "LF": 725.0,       # 정상 성인 LF 평균 (ms²)
            "HF": 975.0,       # 정상 성인 HF 평균 (ms²)
            "LF_HF_ratio": 0.7, # 정상 성인 LF/HF 비율 평균
            "SD1": 35.0,       # 정상 성인 SD1 평균 (ms)
            "SD2": 85.0,       # 정상 성인 SD2 평균 (ms)
        }
        
        # 불안장애 임계값 (특허 및 연구 데이터 기반)
        self.anxiety_thresholds = {
            "SampEn": {
                AnxietyLevel.MILD: 1.35,    # 경미한 불안 임계값
                AnxietyLevel.MODERATE: 1.15, # 중등도 불안 임계값
                AnxietyLevel.SEVERE: 0.95,   # 심각한 불안 임계값
                AnxietyLevel.PANIC: 0.75,    # 패닉 상태 임계값
            },
            "SDNN": {
                AnxietyLevel.MILD: 55.0,
                AnxietyLevel.MODERATE: 45.0,
                AnxietyLevel.SEVERE: 35.0,
                AnxietyLevel.PANIC: 25.0,
            },
            "RMSSD": {
                AnxietyLevel.MILD: 35.0,
                AnxietyLevel.MODERATE: 28.0,
                AnxietyLevel.SEVERE: 20.0,
                AnxietyLevel.PANIC: 15.0,
            },
            "LF_HF_ratio": {
                AnxietyLevel.MILD: 0.9,
                AnxietyLevel.MODERATE: 1.2,
                AnxietyLevel.SEVERE: 1.5,
                AnxietyLevel.PANIC: 2.0,
            }
        }
        
        # 마지막 불안 수준 및 예측 결과 저장
        self.last_prediction = {
            "timestamp": 0,
            "anxiety_level": AnxietyLevel.NORMAL,
            "anxiety_score": 0.0,
            "features": {},
            "likelihood": 0.0,
        }
        
        # 최근 예측 히스토리 (최대 100개 저장)
        self.prediction_history = []
        
    def predict_anxiety(self, hrv_features: Dict) -> Dict:
        """
        HRV 지표를 기반으로 불안 수준 예측
        
        Args:
            hrv_features: HRV 분석 지표 딕셔너리
                필수 키: 'SampEn', 'SDNN', 'RMSSD', 'pNN50', 'LF', 'HF', 'LF_HF_ratio'
                선택 키: 'SD1', 'SD2', 'ApEn', 'DFA_alpha1', 'DFA_alpha2'
                
        Returns:
            예측 결과 딕셔너리:
                - anxiety_level: AnxietyLevel 열거형
                - anxiety_score: 0-100 점수
                - features: 입력 피처
                - likelihood: 불안장애 발생 가능성 (0.0-1.0)
                - timestamp: 타임스탬프
        """
        # 필수 HRV 지표 확인
        required_features = ['SampEn', 'SDNN', 'RMSSD', 'LF_HF_ratio']
        missing = [f for f in required_features if f not in hrv_features]
        
        if missing:
            logger.error(f"Missing required HRV features: {missing}")
            return {
                "error": f"Missing required features: {missing}",
                "anxiety_level": self.last_prediction["anxiety_level"],
                "anxiety_score": self.last_prediction["anxiety_score"],
                "timestamp": int(time.time()),
            }
            
        # 예측 결과 딕셔너리 초기화
        result = {
            "anxiety_level": AnxietyLevel.NORMAL,
            "anxiety_score": 0.0,
            "features": hrv_features,
            "likelihood": 0.0,
            "timestamp": int(time.time()),
        }
        
        # 기계학습 모델 기반 예측 또는 규칙 기반 예측
        if not self.use_rule_based and self.model:
            result = self._predict_with_ml_model(hrv_features)
        else:
            result = self._predict_with_rules(hrv_features)
            
        # 예측 결과 저장
        self.last_prediction = result
        
        # 이력에 추가
        self.prediction_history.append(result)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
            
        return result
        
    def _predict_with_ml_model(self, hrv_features: Dict) -> Dict:
        """
        기계학습 모델 기반 불안 수준 예측
        
        Args:
            hrv_features: HRV 분석 지표 딕셔너리
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 특성 벡터 구성
            feature_names = self.model.feature_names_in_
            X = np.array([[hrv_features.get(f, 0.0) for f in feature_names]])
            
            # 특성 스케일링
            X_scaled = self.scaler.transform(X)
            
            # 예측 수행
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                anxiety_score = proba[1] * 100.0  # 이진 분류에서 양성 클래스의 확률
                likelihood = proba[1]
            else:
                # 확률을 제공하지 않는 모델의 경우
                prediction = self.model.predict(X_scaled)[0]
                anxiety_score = prediction * 100.0  # 연속값 예측 가정
                likelihood = prediction
                
            # 불안 수준 결정
            anxiety_level = self._score_to_anxiety_level(anxiety_score)
            
            return {
                "anxiety_level": anxiety_level,
                "anxiety_score": anxiety_score,
                "features": hrv_features,
                "likelihood": likelihood,
                "timestamp": int(time.time()),
            }
            
        except Exception as e:
            logger.error(f"Error in ML model prediction: {e}")
            # 오류 발생 시 규칙 기반으로 대체
            return self._predict_with_rules(hrv_features)
            
    def _predict_with_rules(self, hrv_features: Dict) -> Dict:
        """
        규칙 기반 불안 수준 예측 (특허 알고리즘 구현)
        
        특허 10-2022-0007209에 명시된 방법에 따라 SampEn 값과
        다른 HRV 지표를 고려하여 불안 수준을 예측합니다.
        
        Args:
            hrv_features: HRV 분석 지표 딕셔너리
            
        Returns:
            예측 결과 딕셔너리
        """
        # SampEn 평가 (특허의 주요 예측 인자)
        samp_en = hrv_features.get('SampEn', 0.0)
        anxiety_level = AnxietyLevel.NORMAL
        
        # SampEn 임계값 기반 초기 분류
        for level, threshold in self.anxiety_thresholds['SampEn'].items():
            if samp_en <= threshold:
                anxiety_level = level
                break
                
        # 다른 HRV 지표 평가
        # 불안장애 상태에서는 SDNN, RMSSD 감소, LF/HF 비율 증가가 일반적
        factor_count = 0
        total_factors = 0
        
        # SDNN 평가
        if 'SDNN' in hrv_features:
            total_factors += 1
            sdnn = hrv_features['SDNN']
            for level, threshold in self.anxiety_thresholds['SDNN'].items():
                if sdnn <= threshold and level.value >= anxiety_level.value:
                    factor_count += 1
                    break
                    
        # RMSSD 평가
        if 'RMSSD' in hrv_features:
            total_factors += 1
            rmssd = hrv_features['RMSSD']
            for level, threshold in self.anxiety_thresholds['RMSSD'].items():
                if rmssd <= threshold and level.value >= anxiety_level.value:
                    factor_count += 1
                    break
                    
        # LF/HF 비율 평가
        if 'LF_HF_ratio' in hrv_features:
            total_factors += 1
            lf_hf = hrv_features['LF_HF_ratio']
            for level, threshold in self.anxiety_thresholds['LF_HF_ratio'].items():
                if lf_hf >= threshold and level.value >= anxiety_level.value:
                    factor_count += 1
                    break
                    
        # 추가 요소들의 확인 비율에 따른 불안 점수 조정
        confirmation_ratio = factor_count / total_factors if total_factors > 0 else 0
        
        # 최종 불안 점수 (0-100) 계산
        base_score = anxiety_level.value * 25  # 각 레벨마다 25점씩 (PANIC은 100점)
        
        # 경계값에서의 점수 조정 (SampEn 기준)
        if anxiety_level != AnxietyLevel.NORMAL:
            curr_threshold = self.anxiety_thresholds['SampEn'][anxiety_level]
            next_level = AnxietyLevel(anxiety_level.value - 1) if anxiety_level.value > 0 else anxiety_level
            next_threshold = self.anxiety_thresholds['SampEn'].get(next_level, self.reference_values['SampEn'])
            
            # SampEn이 임계값들 사이에서 어디에 위치하는지 계산
            position_ratio = 0.0
            if curr_threshold < next_threshold:  # SampEn은 낮을수록 불안
                position_ratio = (next_threshold - samp_en) / (next_threshold - curr_threshold)
            
            # 위치 비율에 따른 점수 조정
            position_ratio = max(0.0, min(1.0, position_ratio))
            base_score -= (1.0 - position_ratio) * 15  # 최대 15점 감소 가능
        
        # 확인 요소들의 비율에 따른 점수 조정
        confidence_factor = 10.0  # 확인 요소 최대 영향 점수
        anxiety_score = base_score + (confirmation_ratio * confidence_factor)
        
        # 최종 불안 점수 제한 (0-100)
        anxiety_score = max(0.0, min(100.0, anxiety_score))
        
        # 점수에 따른 최종 불안 수준 재계산
        final_anxiety_level = self._score_to_anxiety_level(anxiety_score)
        
        # 불안장애 발생 가능성 계산 (0.0-1.0)
        likelihood = anxiety_score / 100.0
        
        return {
            "anxiety_level": final_anxiety_level,
            "anxiety_score": anxiety_score,
            "features": hrv_features,
            "likelihood": likelihood,
            "timestamp": int(time.time()),
        }
        
    def _score_to_anxiety_level(self, score: float) -> AnxietyLevel:
        """
        불안 점수를 불안 수준으로 변환
        
        Args:
            score: 불안 점수 (0-100)
            
        Returns:
            불안 수준 열거형
        """
        if score < 20:
            return AnxietyLevel.NORMAL
        elif score < 40:
            return AnxietyLevel.MILD
        elif score < 60:
            return AnxietyLevel.MODERATE
        elif score < 80:
            return AnxietyLevel.SEVERE
        else:
            return AnxietyLevel.PANIC
            
    def get_trend(self, window: int = 10) -> Dict:
        """
        최근 불안 수준 추세 분석
        
        Args:
            window: 분석 윈도우 크기 (최근 예측 수)
            
        Returns:
            추세 정보 딕셔너리
        """
        if not self.prediction_history:
            return {
                "trend": "stable",
                "slope": 0.0,
                "current": 0.0,
                "prediction": 0.0
            }
            
        # 분석 윈도우 설정
        window = min(window, len(self.prediction_history))
        recent = self.prediction_history[-window:]
        
        # 시간에 따른 불안 점수 추출
        times = np.array([(p["timestamp"] - recent[0]["timestamp"]) / 60 for p in recent])  # 분 단위
        scores = np.array([p["anxiety_score"] for p in recent])
        
        # 선형 회귀로 추세 계산
        if len(times) > 1:
            # 기울기 및 절편 계산
            slope, intercept = np.polyfit(times, scores, 1)
            
            # 현재 값
            current = scores[-1]
            
            # 30분 후 예상 값
            future_time = times[-1] + 30  # 현재 시간 + 30분
            prediction = slope * future_time + intercept
            prediction = max(0.0, min(100.0, prediction))  # 0-100 범위로 제한
            
            # 추세 판정
            if abs(slope) < 0.1:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
                
        else:
            slope = 0.0
            current = scores[0]
            prediction = current
            trend = "stable"
            
        return {
            "trend": trend,
            "slope": slope,
            "current": current,
            "prediction": prediction
        }
        
    def get_intervention_recommendation(self) -> Dict:
        """
        현재 불안 수준에 따른 중재 권장 사항 반환
        
        Returns:
            중재 권장 사항 딕셔너리
        """
        anxiety_level = self.last_prediction["anxiety_level"]
        anxiety_score = self.last_prediction["anxiety_score"]
        trend = self.get_trend()
        
        # 기본 권장 사항 초기화
        recommendation = {
            "intervention_needed": False,
            "stimulation_recommended": False,
            "stimulation_params": {
                "frequency": 30.0,  # Hz
                "intensity": 2.0,   # mA
                "duration": 15,     # minutes
                "waveform": "biphasic",
                "phase_delay": 0.5  # seconds
            },
            "urgency": "none",
            "message": "정상 상태입니다. 지속적인 모니터링을 유지하세요."
        }
        
        # 불안 수준 및 추세에 따른 권장 사항 조정
        if anxiety_level == AnxietyLevel.NORMAL:
            # 정상 상태지만 증가 추세인 경우 주의
            if trend["trend"] == "increasing" and trend["prediction"] > 30:
                recommendation["message"] = "현재는 정상이지만 불안이 증가하는 추세입니다. 호흡 운동을 권장합니다."
                
        elif anxiety_level == AnxietyLevel.MILD:
            recommendation["intervention_needed"] = True
            recommendation["urgency"] = "low"
            recommendation["message"] = "경미한 불안이 감지되었습니다. 심호흡과 간단한 이완 운동을 권장합니다."
            
            # 증가 추세인 경우 추가 주의
            if trend["trend"] == "increasing":
                recommendation["message"] += " 불안이 증가하는 추세이므로 주의가 필요합니다."
                
        elif anxiety_level == AnxietyLevel.MODERATE:
            recommendation["intervention_needed"] = True
            recommendation["stimulation_recommended"] = True
            recommendation["urgency"] = "medium"
            recommendation["message"] = "중등도 불안이 감지되었습니다. 저강도 두개전기자극을 권장합니다."
            
            # 자극 파라미터 조정
            recommendation["stimulation_params"]["frequency"] = 20.0
            recommendation["stimulation_params"]["intensity"] = 1.5
            recommendation["stimulation_params"]["duration"] = 15
            
        elif anxiety_level == AnxietyLevel.SEVERE:
            recommendation["intervention_needed"] = True
            recommendation["stimulation_recommended"] = True
            recommendation["urgency"] = "high"
            recommendation["message"] = "심각한 불안이 감지되었습니다. 중강도 두개전기자극을 즉시 시작하는 것을 권장합니다."
            
            # 자극 파라미터 조정
            recommendation["stimulation_params"]["frequency"] = 30.0
            recommendation["stimulation_params"]["intensity"] = 2.5
            recommendation["stimulation_params"]["duration"] = 20
            recommendation["stimulation_params"]["phase_delay"] = 0.3
            
        elif anxiety_level == AnxietyLevel.PANIC:
            recommendation["intervention_needed"] = True
            recommendation["stimulation_recommended"] = True
            recommendation["urgency"] = "critical"
            recommendation["message"] = "패닉 수준의 불안이 감지되었습니다. 강도 높은 두개전기자극을 즉시 시작하는 것을 권장합니다."
            
            # 자극 파라미터 조정
            recommendation["stimulation_params"]["frequency"] = 40.0
            recommendation["stimulation_params"]["intensity"] = 3.0
            recommendation["stimulation_params"]["duration"] = 25
            recommendation["stimulation_params"]["phase_delay"] = 0.2
            
        return recommendation
        
    def train_model(self, training_data: pd.DataFrame, target_column: str) -> bool:
        """
        HRV 데이터로 불안 예측 모델 학습
        
        Args:
            training_data: HRV 특성 및 불안 수준 라벨이 포함된 데이터프레임
            target_column: 목표 변수 열 이름
            
        Returns:
            학습 성공 여부
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # 특성과 타겟 분리
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 학습
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # 규칙 기반에서 ML 기반으로 전환
            self.use_rule_based = False
            
            logger.info("Successfully trained anxiety prediction model")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
            
    def save_model(self, path: str) -> bool:
        """
        학습된 모델 저장
        
        Args:
            path: 저장 경로
            
        Returns:
            저장 성공 여부
        """
        if not self.model or self.use_rule_based:
            logger.error("No ML model available to save")
            return False
            
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
