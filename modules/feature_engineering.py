"""
OVIP - Feature Engineering Module
Creates all features used by the ML models without data leakage.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates features for ML models with strict data leakage prevention."""
    
    def __init__(self, train_cutoff: str = '2021-01-01'):
        """
        Args:
            train_cutoff: Date to split train/test for safe feature creation
        """
        self.train_cutoff = pd.to_datetime(train_cutoff)
        self.train_stats = {}
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full feature engineering pipeline."""
        df = df.copy()
        
        # Ensure Date is datetime
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Execute pipeline in order
        df = self._create_basic_lags(df)
        df = self._create_regime_features(df)
        df = self._create_volatility_features(df)
        df = self._create_nlp_features(df)
        df = self._create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _create_basic_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic 1-month lagged features to prevent look-ahead bias."""
        if 'Volatility' in df.columns:
            df['L_Vol'] = df['Volatility'].shift(1)
        
        if 'Crisis_Prob' in df.columns:
            df['L_Regime'] = df['Crisis_Prob'].shift(1)
        
        if 'Intensity' in df.columns:
            df['L_Inten'] = df['Intensity'].shift(1)
            
        if 'WTI' in df.columns:
            df['Returns'] = df['WTI'].pct_change()
            df['L_WTI_Ret'] = df['Returns'].shift(1)
            
        if 'gpr' in df.columns:
            df['L_GPR'] = df['gpr'].shift(1)
            
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Regime-based Expanding Volatility (MS_Vol_Safe)."""
        if 'L_Regime' not in df.columns:
            return df
            
        df['Regime_Label'] = (df['L_Regime'] > 0.5).astype(int)
        
        if 'Volatility' in df.columns:
            # Expanding window prevents leakage
            df['L_MS_Vol_Safe'] = df.groupby('Regime_Label')['Volatility']\
                .expanding()\
                .std()\
                .reset_index(level=0, drop=True)\
                .shift(1)
            
            # Fill NaNs with global expanding std
            df['L_MS_Vol_Safe'] = df['L_MS_Vol_Safe'].fillna(
                df['Volatility'].expanding().std().shift(1)
            )
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Acceleration and Vol-of-Vol features."""
        if 'Volatility' not in df.columns:
            return df
            
        df['L_Accel'] = df['Volatility'].shift(1).diff()
        df['L_Vol_Std'] = df['Volatility'].shift(1).rolling(6).std()
        df['L_Vol_MA3'] = df['Volatility'].shift(1).rolling(3).mean()
        df['L_Vol_MA12'] = df['Volatility'].shift(1).rolling(12).mean()
        return df
    
    def _create_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Center Sentiment Scores using ONLY Training Data Mean."""
        if 'Score' not in df.columns:
            return df
            
        df['L_News_Shk'] = df['Score'].shift(1).diff()
        
        # Apply train cutoff
        if 'Date' in df.columns:
            train_mask = df['Date'] < self.train_cutoff
            train_mean_score = df.loc[train_mask, 'Score'].shift(1).mean()
        else:
            train_mean_score = df['Score'].iloc[:len(df)//2].shift(1).mean()
            
        self.train_stats['score_mean'] = train_mean_score
        df['Score_Centered'] = df['Score'].shift(1) - train_mean_score
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced interaction terms."""
        if 'L_Regime' in df.columns and 'Score_Centered' in df.columns:
            df['L_State_S_Safe'] = df['L_Regime'] * df['Score_Centered']
            
        if 'Score_Centered' in df.columns and 'L_Inten' in df.columns:
            df['L_Crowd_Safe'] = df['Score_Centered'] * np.log1p(df['L_Inten'])
            
        return df
    
    def get_nprs1_features(self, df: pd.DataFrame) -> List[str]:
        """Returns the 5 features required for the NPRS-1 Direction Model."""
        features = ['L_Vol', 'L_Regime', 'L_Inten', 'L_GPR', 'L_Accel']
        return [f for f in features if f in df.columns]
    
    def get_rf11_features(self, df: pd.DataFrame) -> List[str]:
        """Returns the 11 features required for the RF-11 Level Model."""
        features = [
            'L_Vol', 'L_Regime', 'L_Inten', 'L_WTI_Ret', 'L_GPR',
            'L_Accel', 'L_News_Shk', 'L_MS_Vol_Safe',
            'L_State_S_Safe', 'L_Crowd_Safe', 'L_Vol_Std'
        ]
        return [f for f in features if f in df.columns]
    
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the 1/0 target for the Direction Classifier."""
        if 'Volatility' in df.columns:
            df['Vol_Direction'] = (df['Volatility'] > df['Volatility'].shift(1)).astype(int)
        return df

    def validate_features(self, df: pd.DataFrame, feature_list: List[str]) -> bool:
        """Checks for missing columns or NaN values in the Test Set."""
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            logger.error(f"Missing features: {missing}")
            return False
            
        if 'Date' in df.columns:
            test_mask = df['Date'] >= self.train_cutoff
            test_data = df.loc[test_mask, feature_list]
        else:
            test_data = df.iloc[len(df)//2:][feature_list]
            
        nan_counts = test_data.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values found in TEST SET:\n{nan_counts[nan_counts > 0]}")
            return False
            
        return True

if __name__ == '__main__':
    # Simple Local Test utilizing the DataLoader we built earlier
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.data_loader import get_data_loader
    
    print("Testing Feature Engineering Pipeline...")
    
    loader = get_data_loader()
    df_raw = loader.merge_all_data()
    
    if not df_raw.empty:
        engineer = FeatureEngineer(train_cutoff='2021-01-01')
        df_engineered = engineer.create_all_features(df_raw)
        df_engineered = engineer.create_binary_target(df_engineered)
        
        rf11 = engineer.get_rf11_features(df_engineered)
        is_valid = engineer.validate_features(df_engineered, rf11)
        
        print(f"Features Generated Successfully. Final shape: {df_engineered.shape}")
        print(f"Test Set Validation Passed: {is_valid}")
    else:
        print("Could not load data. Ensure merged_final.csv is in the /data folder.")
