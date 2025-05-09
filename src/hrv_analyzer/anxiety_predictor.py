"""
Anxiety Prediction Model 

This module provides functionality to predict anxiety levels based on HRV parameters,
as implemented in the patent 10-2022-0007209 ("불안장애 예방장치").

The model analyzes various HRV parameters to determine the likelihood of anxiety disorders 
and classifies the current state into different risk levels.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from typing import Dict, List, Tuple, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define anxiety risk levels
ANXIETY_RISK_LEVELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Very High"
}

class AnxietyPredictor:
    """
    A class for predicting anxiety levels based on HRV parameters.
    
    This predictor uses a machine learning model trained on HRV data to 
    estimate the likelihood of anxiety disorders and classify the current 
    state into different risk levels.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the anxiety predictor.
        
        Args:
            model_path: Path to a pre-trained model file. If None, a new model 
                       will be created and trained with sample data.
        """
        self.scaler = StandardScaler()
        
        # Important HRV features for anxiety prediction based on research
        self.feature_names = [
            'SDNN',         # Standard deviation of NN intervals
            'RMSSD',        # Root mean square of successive RR interval differences
            'pNN50',        # Proportion of NN50 divided by total number of NNs
            'LF_power',     # Low-frequency power
            'HF_power',     # High-frequency power
            'LF_HF_ratio',  # LF/HF ratio
            'SD1',          # Poincaré plot standard deviation perpendicular to the line of identity
            'SD2',          # Poincaré plot standard deviation along the line of identity
            'SampEn'        # Sample entropy
        ]
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"Loaded anxiety prediction model from {model_path}")
        else:
            self._create_model()
            logger.info("Created new anxiety prediction model")
    
    def _create_model(self):
        """Create and train a new model with sample data."""
        # Create a Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train with sample data (this would be replaced with real training data)
        X_sample, y_sample = self._generate_sample_data()
        
        # Fit the scaler
        self.scaler.fit(X_sample)
        
        # Scale the features
        X_scaled = self.scaler.transform(X_sample)
        
        # Train the model
        self.model.fit(X_scaled, y_sample)
        
    def _load_model(self, model_path: str):
        """
        Load a pre-trained model from a file.
        
        Args:
            model_path: Path to the model file.
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
    def save_model(self, model_path: str):
        """
        Save the trained model to a file.
        
        Args:
            model_path: Path where the model will be saved.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def predict(self, hrv_params: Dict[str, float]) -> Dict[str, Union[int, str, float]]:
        """
        Predict anxiety risk level based on HRV parameters.
        
        Args:
            hrv_params: Dictionary containing HRV parameters.
                       Should include values for the features defined in self.feature_names.
        
        Returns:
            Dictionary containing the predicted risk level (0-3), risk category (Low to Very High),
            and confidence score.
        """
        # Extract features from the parameters
        features = []
        for feature in self.feature_names:
            if feature in hrv_params:
                features.append(hrv_params[feature])
            else:
                logger.warning(f"Missing feature: {feature}")
                features.append(0)  # Default value if feature is missing
                
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        # Get predicted class
        risk_level = self.model.predict(scaled_features)[0]
        
        # Get confidence score (probability of the predicted class)
        confidence = probabilities[risk_level]
        
        # Return prediction results
        return {
            'risk_level': int(risk_level),
            'risk_category': ANXIETY_RISK_LEVELS[risk_level],
            'confidence': float(confidence),
            'probabilities': {f"Level_{i}": float(prob) for i, prob in enumerate(probabilities)}
        }
    
    def _generate_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data for training the model.
        
        This is used only for demonstration purposes. In a real implementation,
        the model would be trained on actual HRV data from people with and without
        anxiety disorders.
        
        Returns:
            X_sample: Sample feature matrix.
            y_sample: Sample target vector (anxiety risk levels).
        """
        # Sample size
        n_samples = 500
        
        # Create DataFrame to store sample data
        data = []
        
        # Generate data for each anxiety level (0-3)
        for level in range(4):
            # Number of samples for this level
            level_samples = n_samples // 4
            
            # Base values and variations for each level
            if level == 0:  # Low anxiety
                base_values = {
                    'SDNN': 50,        # Higher SDNN indicates better ANS function
                    'RMSSD': 40,       # Higher RMSSD indicates better parasympathetic activity
                    'pNN50': 20,       # Higher pNN50 indicates better parasympathetic activity
                    'LF_power': 1500,  # Moderate LF power
                    'HF_power': 1700,  # Higher HF power indicates better parasympathetic activity
                    'LF_HF_ratio': 0.9, # Lower ratio indicates parasympathetic dominance
                    'SD1': 30,         # Higher SD1 indicates better short-term variability
                    'SD2': 70,         # Higher SD2 indicates better long-term variability
                    'SampEn': 1.7      # Higher SampEn indicates better complexity/adaptability
                }
                variations = {
                    'SDNN': 10,
                    'RMSSD': 10,
                    'pNN50': 5,
                    'LF_power': 300,
                    'HF_power': 300,
                    'LF_HF_ratio': 0.2,
                    'SD1': 7,
                    'SD2': 15,
                    'SampEn': 0.3
                }
            elif level == 1:  # Moderate anxiety
                base_values = {
                    'SDNN': 40,
                    'RMSSD': 30,
                    'pNN50': 15,
                    'LF_power': 1700,
                    'HF_power': 1300,
                    'LF_HF_ratio': 1.3,
                    'SD1': 25,
                    'SD2': 60,
                    'SampEn': 1.4
                }
                variations = {
                    'SDNN': 8,
                    'RMSSD': 8,
                    'pNN50': 4,
                    'LF_power': 300,
                    'HF_power': 250,
                    'LF_HF_ratio': 0.2,
                    'SD1': 5,
                    'SD2': 12,
                    'SampEn': 0.25
                }
            elif level == 2:  # High anxiety
                base_values = {
                    'SDNN': 30,
                    'RMSSD': 20,
                    'pNN50': 10,
                    'LF_power': 1900,
                    'HF_power': 900,
                    'LF_HF_ratio': 2.1,
                    'SD1': 18,
                    'SD2': 45,
                    'SampEn': 1.1
                }
                variations = {
                    'SDNN': 7,
                    'RMSSD': 6,
                    'pNN50': 3,
                    'LF_power': 250,
                    'HF_power': 200,
                    'LF_HF_ratio': 0.3,
                    'SD1': 4,
                    'SD2': 10,
                    'SampEn': 0.2
                }
            else:  # Very high anxiety
                base_values = {
                    'SDNN': 20,
                    'RMSSD': 12,
                    'pNN50': 5,
                    'LF_power': 2100,
                    'HF_power': 600,
                    'LF_HF_ratio': 3.5,
                    'SD1': 10,
                    'SD2': 30,
                    'SampEn': 0.8
                }
                variations = {
                    'SDNN': 5,
                    'RMSSD': 4,
                    'pNN50': 2,
                    'LF_power': 200,
                    'HF_power': 150,
                    'LF_HF_ratio': 0.4,
                    'SD1': 3,
                    'SD2': 8,
                    'SampEn': 0.15
                }
            
            # Generate samples for this level
            for _ in range(level_samples):
                sample = {}
                for feature in self.feature_names:
                    base = base_values[feature]
                    variation = variations[feature]
                    # Generate value with normal distribution around base
                    sample[feature] = max(0, np.random.normal(base, variation))
                
                # Add label (anxiety level)
                sample['anxiety_level'] = level
                
                # Add to data
                data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Extract features and target
        X = df[self.feature_names].values
        y = df['anxiety_level'].values
        
        return X, y
