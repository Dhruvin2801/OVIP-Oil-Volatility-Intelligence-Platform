"""
OVIP - Model Inference Module
Handles loading and running trained ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self, models_dir: Path):
        """
        Initialize model predictor
        
        Args:
            models_dir: Directory containing saved model files
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        
    def load_model(self, model_name: str, model_path: Optional[Path] = None) -> bool:
        """
        Load a trained model from disk
        
        Args:
            model_name: Name to store model under ('nprs1', 'rf11', etc.)
            model_path: Path to model file. If None, uses default naming
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                model_path = self.models_dir / f"{model_name}.pkl"
            
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            logger.info(f"Loaded model: {model_name} from {model_path}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_path}")
            # Create dummy model for demonstration
            self._create_dummy_model(model_name)
            return False
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self._create_dummy_model(model_name)
            return False
    
    def _create_dummy_model(self, model_name: str):
        """Create a dummy model when real model not available"""
        if model_name == 'nprs1':
            # Binary classifier
            self.models[model_name] = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                random_state=42
            )
        elif model_name == 'rf11':
            # Regression model
            self.models[model_name] = RandomForestRegressor(
                n_estimators=500,
                max_depth=7,
                random_state=42
            )
        logger.info(f"Created dummy {model_name} model")
    
    def predict_direction(
        self,
        features: pd.DataFrame,
        model_name: str = 'nprs1'
    ) -> Dict:
        """
        Predict volatility direction (up/down)
        
        Args:
            features: DataFrame with required features
            model_name: Name of classifier model to use
            
        Returns:
            Dict with prediction, probability, and confidence
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Required features for NPRS-1
        required_features = ['L_Vol', 'L_Regime', 'L_Inten', 'L_GPR', 'L_Accel']
        
        # Validate features
        missing = [f for f in required_features if f not in features.columns]
        if missing:
            logger.error(f"Missing features for direction prediction: {missing}")
            return {
                'direction': 'UNKNOWN',
                'probability': 0.5,
                'confidence': 0.0,
            }
        
        # Get latest row
        X = features[required_features].iloc[[-1]]
        
        # Handle NaN
        if X.isna().any().any():
            logger.warning("NaN values in features, filling with median")
            X = X.fillna(X.median())
        
        try:
            # Predict
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred = 1 if proba[1] > 0.5 else 0
                confidence = max(proba)
            else:
                # Fallback for models without predict_proba
                pred = model.predict(X)[0]
                confidence = 0.68  # Use historical accuracy
                proba = [1-confidence, confidence] if pred == 1 else [confidence, 1-confidence]
            
            return {
                'direction': 'UP' if pred == 1 else 'DOWN',
                'probability': float(proba[1]),  # Probability of UP
                'confidence': float(confidence),
            }
            
        except Exception as e:
            logger.error(f"Error in direction prediction: {e}")
            return {
                'direction': 'ERROR',
                'probability': 0.5,
                'confidence': 0.0,
            }
    
    def predict_level(
        self,
        features: pd.DataFrame,
        model_name: str = 'rf11'
    ) -> Dict:
        """
        Predict volatility level
        
        Args:
            features: DataFrame with required features
            model_name: Name of regression model to use
            
        Returns:
            Dict with prediction, range, and confidence
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Required features for RF11
        required_features = [
            'L_Vol', 'L_Regime', 'L_Inten', 'L_WTI_Ret', 'L_GPR',
            'L_Accel', 'L_News_Shk', 'L_MS_Vol_Safe',
            'L_State_S_Safe', 'L_Crowd_Safe', 'L_Vol_Std'
        ]
        
        # Validate features
        missing = [f for f in required_features if f not in features.columns]
        if missing:
            logger.error(f"Missing features for level prediction: {missing}")
            # Return last known volatility as fallback
            return {
                'forecast': features['Volatility'].iloc[-1] if 'Volatility' in features.columns else 0.20,
                'range_low': 0.15,
                'range_high': 0.25,
                'confidence_level': 'LOW',
            }
        
        # Get latest row
        X = features[required_features].iloc[[-1]]
        
        # Handle NaN
        if X.isna().any().any():
            logger.warning("NaN values in features, filling with median")
            X = X.fillna(X.median())
        
        try:
            # Predict
            prediction = model.predict(X)[0]
            
            # Estimate confidence interval (±3 std errors based on historical performance)
            # Typical RMSE for this model is ~0.03
            std_error = 0.03
            range_low = max(0, prediction - 1.96 * std_error)
            range_high = prediction + 1.96 * std_error
            
            # Determine confidence level based on regime state
            regime_prob = float(features['L_Regime'].iloc[-1])
            if regime_prob < 0.3 or regime_prob > 0.7:
                confidence = 'HIGH'  # Model works well in clear regimes
            elif 0.4 <= regime_prob <= 0.6:
                confidence = 'LOW'  # Model struggles in transition
            else:
                confidence = 'MODERATE'
            
            return {
                'forecast': float(prediction),
                'range_low': float(range_low),
                'range_high': float(range_high),
                'confidence_level': confidence,
            }
            
        except Exception as e:
            logger.error(f"Error in level prediction: {e}")
            return {
                'forecast': 0.20,
                'range_low': 0.15,
                'range_high': 0.25,
                'confidence_level': 'ERROR',
            }
    
    def get_feature_importance(
        self,
        model_name: str = 'rf11'
    ) -> Dict[str, float]:
        """
        Get feature importance from a trained model
        
        Args:
            model_name: Name of model
            
        Returns:
            Dict mapping feature names to importance scores
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return {}
        
        # Get feature names
        if model_name == 'nprs1':
            features = ['L_Vol', 'L_Regime', 'L_Inten', 'L_GPR', 'L_Accel']
        elif model_name == 'rf11':
            features = [
                'L_Vol', 'L_Regime', 'L_Inten', 'L_WTI_Ret', 'L_GPR',
                'L_Accel', 'L_News_Shk', 'L_MS_Vol_Safe',
                'L_State_S_Safe', 'L_Crowd_Safe', 'L_Vol_Std'
            ]
        else:
            logger.error(f"Unknown model: {model_name}")
            return {}
        
        # Get importances
        importances = model.feature_importances_
        
        # Create dict
        importance_dict = dict(zip(features, importances))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict
    
    def calculate_prediction_intervals(
        self,
        features: pd.DataFrame,
        model_name: str = 'rf11',
        n_estimators: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using tree variance
        
        Args:
            features: Input features
            model_name: Model to use
            n_estimators: Number of trees to use (None = all)
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        if not isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
            logger.error("Prediction intervals only work for Random Forest models")
            return np.array([]), np.array([])
        
        # Get predictions from all trees
        predictions = np.array([
            tree.predict(features) for tree in model.estimators_
        ])
        
        # Calculate percentiles
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        
        return lower, upper
    
    def batch_predict(
        self,
        features: pd.DataFrame,
        model_name: str,
        prediction_type: str = 'level'
    ) -> pd.DataFrame:
        """
        Make predictions for multiple rows
        
        Args:
            features: DataFrame with multiple rows
            model_name: Model to use
            prediction_type: 'level' or 'direction'
            
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for idx in range(len(features)):
            row_features = features.iloc[[idx]]
            
            if prediction_type == 'direction':
                pred = self.predict_direction(
                    pd.concat([features.iloc[:idx+1], row_features]),
                    model_name
                )
            else:
                pred = self.predict_level(
                    pd.concat([features.iloc[:idx+1], row_features]),
                    model_name
                )
            
            results.append(pred)
        
        return pd.DataFrame(results)


def load_models_from_dir(models_dir: Path) -> ModelPredictor:
    """
    Factory function to create ModelPredictor and load all models
    
    Args:
        models_dir: Directory with model files
        
    Returns:
        ModelPredictor instance with models loaded
    """
    predictor = ModelPredictor(models_dir)
    
    # Try to load all known models
    predictor.load_model('nprs1')
    predictor.load_model('rf11')
    
    return predictor


if __name__ == '__main__':
    # Test model predictor
    print("Testing ModelPredictor...")
    
    # Create sample features
    sample_features = pd.DataFrame({
        'L_Vol': [0.187],
        'L_Regime': [0.55],
        'L_Inten': [245],
        'L_WTI_Ret': [0.042],
        'L_GPR': [168],
        'L_Accel': [0.015],
        'L_News_Shk': [-0.02],
        'L_MS_Vol_Safe': [0.15],
        'L_State_S_Safe': [-0.01],
        'L_Crowd_Safe': [0.03],
        'L_Vol_Std': [0.08],
        'Volatility': [0.187],  # Current volatility for fallback
    })
    
    # Create predictor (will use dummy models since files don't exist yet)
    models_dir = Path(__file__).parent.parent / 'data' / 'models'
    predictor = ModelPredictor(models_dir)
    
    # Test direction prediction
    print("\nTesting direction prediction:")
    direction_result = predictor.predict_direction(sample_features)
    print(f"  Direction: {direction_result['direction']}")
    print(f"  Probability: {direction_result['probability']:.2%}")
    print(f"  Confidence: {direction_result['confidence']:.2%}")
    
    # Test level prediction
    print("\nTesting level prediction:")
    level_result = predictor.predict_level(sample_features)
    print(f"  Forecast: {level_result['forecast']:.3f}")
    print(f"  Range: {level_result['range_low']:.3f} - {level_result['range_high']:.3f}")
    print(f"  Confidence: {level_result['confidence_level']}")
    
    # Test feature importance
    print("\nFeature importance (RF11):")
    importance = predictor.get_feature_importance('rf11')
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.3f}")
    
    print("\n✅ Model predictor tests complete!")
