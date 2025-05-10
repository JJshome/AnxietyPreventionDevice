import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnxietyPredictor')

class AnxietyPredictor:
    """
    특허 10-2022-0007209에 따른 불안장애 예측 모델
    
    HRV 특성을 기반으로 불안장애 발생 가능성을 예측합니다.
    """
    
    def __init__(self, model_type='random_forest', model_path=None, threshold=0.6):
        """
        AnxietyPredictor 초기화
        
        Args:
            model_type (str): 예측 모델 유형 ('random_forest', 'gradient_boosting', 'svm')
            model_path (str): 미리 학습된 모델 파일 경로 (없으면 새 모델 생성)
            threshold (float): 불안 상태 판단 임계값 (0-1)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # 기본 HRV 특성 목록 (모델 학습에 사용)
        self.default_features = [
            'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'hr', 'lf_power', 'hf_power', 
            'lf_hf_ratio', 'lf_nu', 'hf_nu', 'sd1', 'sd2', 'sd_ratio', 'sampen'
        ]
        
        # 모델 로드 시도
        if model_path:
            try:
                self._load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                self._create_model()
        else:
            self._create_model()
    
    def _create_model(self):
        """
        예측 모델 생성
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                min_samples_split=5, min_samples_leaf=2, random_state=42
            )
        elif self.model_type == 'svm':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
            ])
        else:
            logger.warning(f"Unknown model type: {self.model_type}, fallback to random_forest")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        logger.info(f"Created {self.model_type} model")
    
    def _load_model(self, model_path):
        """
        저장된 모델 로드
        
        Args:
            model_path (str): 모델 파일 경로
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names', self.default_features)
        self.scaler = model_data.get('scaler', StandardScaler())
        self.threshold = model_data.get('threshold', self.threshold)
    
    def save_model(self, model_path):
        """
        모델 저장
        
        Args:
            model_path (str): 저장할 모델 파일 경로
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'threshold': self.threshold
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def train(self, X, y, feature_names=None, test_size=0.2, optimize=False):
        """
        불안장애 예측 모델 학습
        
        Args:
            X (np.ndarray): 입력 특성 배열
            y (np.ndarray): 레이블 배열 (0: 정상, 1: 불안)
            feature_names (list): 특성 이름 목록
            test_size (float): 테스트 세트 비율
            optimize (bool): 하이퍼파라미터 최적화 수행 여부
            
        Returns:
            dict: 모델 평가 결과
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = self.default_features
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 하이퍼파라미터 최적화
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            self._optimize_hyperparameters(X_train_scaled, y_train)
        
        # 모델 학습
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # 모델 평가
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 평가 지표
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # 특성 중요도 (모델 유형에 따라)
        if hasattr(self.model, 'feature_importances_'):
            evaluation['feature_importances'] = dict(zip(
                self.feature_names, self.model.feature_importances_
            ))
        elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps.get('svm', None), 'coef_'):
            evaluation['feature_importances'] = dict(zip(
                self.feature_names, abs(self.model.named_steps['svm'].coef_[0])
            ))
        
        logger.info(f"Model trained with accuracy: {evaluation['accuracy']:.4f}, "
                   f"ROC AUC: {evaluation['roc_auc']:.4f}")
        
        return evaluation
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """
        그리드 서치를 통한 하이퍼파라미터 최적화
        
        Args:
            X_train (np.ndarray): 학습 데이터 특성
            y_train (np.ndarray): 학습 데이터 레이블
        """
        param_grid = {}
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'svm__C': [0.1, 1.0, 10.0],
                'svm__gamma': ['scale', 'auto', 0.1, 0.01],
                'svm__kernel': ['rbf', 'linear']
            }
        
        if not param_grid:
            logger.warning("No parameter grid defined for model optimization")
            return
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best ROC AUC: {grid_search.best_score_:.4f}")
        
        # 최적 모델 적용
        self.model = grid_search.best_estimator_
    
    def predict(self, hrv_features, return_probability=False):
        """
        HRV 특성으로부터 불안장애 발생 가능성 예측
        
        Args:
            hrv_features (dict): HRV 특성 딕셔너리
            return_probability (bool): 확률값 반환 여부
            
        Returns:
            float or tuple: 불안 점수 또는 (예측 레이블, 불안 점수) 튜플
        """
        if not self.model:
            logger.error("Model not initialized or trained")
            return 0.0 if return_probability else (0, 0.0)
        
        # 입력 특성 벡터 생성
        feature_vector = self._prepare_feature_vector(hrv_features)
        
        # 특성 벡터가 비어있는 경우
        if feature_vector.size == 0:
            logger.warning("Empty feature vector for prediction")
            return 0.0 if return_probability else (0, 0.0)
        
        # 특성 스케일링
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # 예측 확률
        anxiety_probability = self.model.predict_proba(feature_vector_scaled)[0, 1]
        
        # 임계값에 따른 레이블
        predicted_label = 1 if anxiety_probability >= self.threshold else 0
        
        if return_probability:
            return anxiety_probability
        else:
            return predicted_label, anxiety_probability
    
    def _prepare_feature_vector(self, hrv_features):
        """
        HRV 특성 딕셔너리에서 모델용 특성 벡터 생성
        
        Args:
            hrv_features (dict): HRV 특성 딕셔너리
            
        Returns:
            np.ndarray: 특성 벡터
        """
        # 특성 이름이 정의되지 않은 경우 기본 특성 사용
        if self.feature_names is None:
            self.feature_names = self.default_features
        
        # 특성 값 추출
        feature_vector = []
        for feature in self.feature_names:
            # 특성이 없는 경우 0으로 대체
            value = hrv_features.get(feature, 0.0)
            feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def set_threshold(self, threshold):
        """
        불안 상태 판단 임계값 설정
        
        Args:
            threshold (float): 새 임계값 (0-1)
        """
        if 0 <= threshold <= 1:
            self.threshold = threshold
            logger.info(f"Anxiety threshold set to {threshold}")
        else:
            logger.warning(f"Invalid threshold value: {threshold}, must be between 0 and 1")
    
    def find_optimal_threshold(self, X_val, y_val, metric='f1'):
        """
        최적의 임계값 찾기
        
        Args:
            X_val (np.ndarray): 검증 데이터 특성
            y_val (np.ndarray): 검증 데이터 레이블
            metric (str): 최적화할 지표 ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            float: 최적 임계값
        """
        if not self.model:
            logger.error("Model not initialized or trained")
            return 0.5
        
        # 특성 스케일링
        X_val_scaled = self.scaler.transform(X_val)
        
        # 예측 확률
        y_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # 다양한 임계값 시도
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            else:  # accuracy
                score = accuracy_score(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold found: {best_threshold:.2f} with {metric}: {best_score:.4f}")
        self.threshold = best_threshold
        
        return best_threshold
    
    def analyze_anxiety_trend(self, hrv_features_list, window_size=5):
        """
        연속적인 HRV 특성으로부터 불안 경향 분석
        
        Args:
            hrv_features_list (list): HRV 특성 딕셔너리 목록
            window_size (int): 이동 평균 윈도우 크기
            
        Returns:
            dict: 불안 경향 분석 결과
        """
        # 빈 입력 처리
        if not hrv_features_list:
            logger.warning("Empty HRV features list for trend analysis")
            return {
                'anxiety_scores': [],
                'trend': 'unknown',
                'peak_score': 0.0,
                'avg_score': 0.0,
                'prediction': 'normal'
            }
        
        # 각 HRV 특성에 대한 불안 점수 계산
        anxiety_scores = [self.predict(features, return_probability=True) for features in hrv_features_list]
        
        # 이동 평균 계산
        if len(anxiety_scores) >= window_size:
            avg_scores = [np.mean(anxiety_scores[i:i+window_size]) 
                          for i in range(len(anxiety_scores) - window_size + 1)]
        else:
            avg_scores = [np.mean(anxiety_scores)]
        
        # 경향 분석
        if len(avg_scores) > 1:
            # 선형 회귀로 경향 기울기 계산
            x = np.arange(len(avg_scores))
            y = np.array(avg_scores)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # 경향 판단
            if m > 0.01:
                trend = 'increasing'
            elif m < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
            m = 0.0
        
        # 최고 점수와 평균 점수
        peak_score = max(anxiety_scores)
        avg_score = np.mean(anxiety_scores)
        
        # 최종 예측
        if peak_score >= self.threshold:
            prediction = 'anxiety'
        elif avg_score >= self.threshold * 0.8:
            prediction = 'potential_anxiety'
        else:
            prediction = 'normal'
        
        # 분석 결과
        result = {
            'anxiety_scores': anxiety_scores,
            'moving_avg_scores': avg_scores,
            'trend': trend,
            'trend_slope': m,
            'peak_score': peak_score,
            'avg_score': avg_score,
            'prediction': prediction
        }
        
        logger.info(f"Anxiety trend analysis: {trend} trend, peak: {peak_score:.4f}, "
                   f"avg: {avg_score:.4f}, prediction: {prediction}")
        
        return result
    
    def get_anxiety_level(self, anxiety_score):
        """
        불안 점수에 따른 불안 수준 반환
        
        Args:
            anxiety_score (float): 불안 점수 (0-1)
            
        Returns:
            str: 불안 수준 ('normal', 'mild', 'moderate', 'severe')
        """
        if anxiety_score < 0.3:
            return 'normal'
        elif anxiety_score < 0.6:
            return 'mild'
        elif anxiety_score < 0.8:
            return 'moderate'
        else:
            return 'severe'