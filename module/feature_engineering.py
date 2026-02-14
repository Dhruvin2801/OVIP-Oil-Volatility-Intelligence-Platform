"""
OVIP - Feature Engineering Module
Creates all features used by the ML models
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for ML models with no data leakage"""
    
    def __init__(self, train_cutoff: str = '2021-01-01'):
        """
        Initialize feature engineer
        
        Args:
            train_cutoff: Date to split train/test for safe feature creation
        """
        self.train_cutoff = pd.to_datetime(train_cutoff)
        self.train_stats = {}
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the models
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Ensure Date is datetime
        if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Create features in order (some depend on others)
        df = self._create_basic_lags(df)
        df = self._create_regime_features(df)
        df = self._create_volatility_features(df)
        df = self._create_nlp_features(df)
        df = self._create_interaction_features(df)
        
        logger.info(f"Created features. Final shape: {df.shape}")
        
        return df
    
    def _create_basic_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic lagged features"""
        
        # L_Vol - Lagged volatility
        if 'Volatility' in df.columns:
            df['L_Vol'] = df['Volatility'].shift(1)
        
        # L_Regime - Lagged crisis probability
        if 'Crisis_Prob' in df.columns:
            df['L_Regime'] = df['Crisis_Prob'].shift(1)
        
        # L_Inten - Lagged news intensity
        if 'Intensity' in df.columns:
            df['L_Inten'] = df['Intensity'].shift(1)
        
        # L_WTI_Ret - Lagged returns
        if 'WTI' in df.columns:
            df['Returns'] = df['WTI'].pct_change()
            df['L_WTI_Ret'] = df['Returns'].shift(1)
        
        # L_GPR - Lagged geopolitical risk
        if 'gpr' in df.columns:
            df['L_GPR'] = df['gpr'].shift(1)
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime-based features"""
        
        if 'L_Regime' not in df.columns:
            return df
        
        # Regime label (binary: 0=calm, 1=crisis)
        df['Regime_Label'] = (df['L_Regime'] > 0.5).astype(int)
        
        # L_MS_Vol_Safe - Markov-switching volatility (expanding window)
        # This calculates std of volatility within each regime state
        if 'Volatility' in df.columns:
            # Use expanding window to avoid future leakage
            df['L_MS_Vol_Safe'] = df.groupby('Regime_Label')['Volatility']\
                .expanding()\
                .std()\
                .reset_index(level=0, drop=True)\
                .shift(1)
            
            # Fill NaN with overall expanding std
            df['L_MS_Vol_Safe'] = df['L_MS_Vol_Safe'].fillna(
                df['Volatility'].expanding().std().shift(1)
            )
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        
        if 'Volatility' not in df.columns:
            return df
        
        # L_Accel - Volatility acceleration (change in volatility)
        df['L_Accel'] = df['Volatility'].shift(1).diff()
        
        # L_Vol_Std - Volatility of volatility (6-month rolling std)
        df['L_Vol_Std'] = df['Volatility'].shift(1).rolling(6).std()
        
        # Additional volatility features
        df['L_Vol_MA3'] = df['Volatility'].shift(1).rolling(3).mean()
        df['L_Vol_MA12'] = df['Volatility'].shift(1).rolling(12).mean()
        
        return df
    
    def _create_nlp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NLP sentiment features"""
        
        if 'Score' not in df.columns:
            return df
        
        # L_News_Shk - News sentiment shock (change in sentiment)
        df['L_News_Shk'] = df['Score'].shift(1).diff()
        
        # Safe centering of sentiment using ONLY train data
        train_mask = df['Date'] < self.train_cutoff if 'Date' in df.columns else slice(None, len(df)//2)
        
        if isinstance(train_mask, pd.Series):
            train_mean_score = df.loc[train_mask, 'Score'].shift(1).mean()
        else:
            train_mean_score = df.iloc[train_mask]['Score'].shift(1).mean()
        
        # Store for later use
        self.train_stats['score_mean'] = train_mean_score
        
        # Centered score (subtract train mean only)
        df['Score_Centered'] = df['Score'].shift(1) - train_mean_score
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        
        # L_State_S_Safe - Regime × Sentiment interaction
        if 'L_Regime' in df.columns and 'Score_Centered' in df.columns:
            df['L_State_S_Safe'] = df['L_Regime'] * df['Score_Centered']
        
        # L_Crowd_Safe - Sentiment × Log(Intensity) interaction
        if 'Score_Centered' in df.columns and 'L_Inten' in df.columns:
            df['L_Crowd_Safe'] = df['Score_Centered'] * np.log1p(df['L_Inten'])
        
        return df
    
    def get_nprs1_features(self, df: pd.DataFrame) -> List[str]:
        """Get feature list for NPRS-1 binary classifier"""
        features = ['L_Vol', 'L_Regime', 'L_Inten', 'L_GPR', 'L_Accel']
        
        # Verify all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Missing NPRS-1 features: {missing}")
        
        return [f for f in features if f in df.columns]
    
    def get_rf11_features(self, df: pd.DataFrame) -> List[str]:
        """Get feature list for 11-Pillar regression model"""
        features = [
            'L_Vol', 'L_Regime', 'L_Inten', 'L_WTI_Ret', 'L_GPR',
            'L_Accel', 'L_News_Shk', 'L_MS_Vol_Safe',
            'L_State_S_Safe', 'L_Crowd_Safe', 'L_Vol_Std'
        ]
        
        # Verify all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Missing RF11 features: {missing}")
        
        return [f for f in features if f in df.columns]
    
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target for direction classification
        
        Args:
            df: DataFrame with Volatility column
            
        Returns:
            DataFrame with Vol_Direction column (1=up, 0=down)
        """
        if 'Volatility' not in df.columns:
            logger.error("Volatility column not found")
            return df
        
        # Direction: 1 if volatility increased, 0 if decreased
        df['Vol_Direction'] = (df['Volatility'] > df['Volatility'].shift(1)).astype(int)
        
        return df
    
    def validate_features(self, df: pd.DataFrame, feature_list: List[str]) -> bool:
        """
        Validate that all required features exist and have no NaN in test set
        
        Args:
            df: DataFrame to validate
            feature_list: List of required features
            
        Returns:
            True if valid, False otherwise
        """
        # Check existence
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            logger.error(f"Missing features: {missing}")
            return False
        
        # Check for NaN in test set
        test_mask = df['Date'] >= self.train_cutoff if 'Date' in df.columns else slice(len(df)//2, None)
        
        if isinstance(test_mask, pd.Series):
            test_data = df.loc[test_mask, feature_list]
        else:
            test_data = df.iloc[test_mask][feature_list]
        
        nan_counts = test_data.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values in test set:\n{nan_counts[nan_counts > 0]}")
            return False
        
        return True
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all engineered features
        
        Args:
            df: DataFrame with features
            
        Returns:
            Summary DataFrame
        """
        # Get all engineered features (those starting with L_)
        feature_cols = [col for col in df.columns if col.startswith('L_')]
        
        if not feature_cols:
            return pd.DataFrame()
        
        summary = df[feature_cols].describe().T
        summary['missing'] = df[feature_cols].isna().sum()
        summary['missing_pct'] = (summary['missing'] / len(df)) * 100
        
        return summary


def create_features_from_performance_data(df_perf: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from model performance CSV
    
    Args:
        df_perf: DataFrame from data_model_performance_2025.csv
        
    Returns:
        DataFrame with features added
    """
    df = df_perf.copy()
    
    # We already have Uncertainty_Factor and L_Regime from the file
    # Add other derived features
    
    # Volatility features
    df['L_Vol'] = df['Volatility'].shift(1)
    df['L_Accel'] = df['Volatility'].shift(1).diff()
    df['L_Vol_Std'] = df['Volatility'].shift(1).rolling(6).std()
    
    # For features not in the file, create dummy versions
    # In production, you'd load these from other data sources
    
    # Dummy NLP features (would come from actual sentiment analysis)
    df['L_Inten'] = 200 + np.random.randn(len(df)) * 50  # News intensity
    df['L_News_Shk'] = np.random.randn(len(df)) * 0.05  # Sentiment shock
    df['Score_Centered'] = np.random.randn(len(df)) * 0.2  # Centered sentiment
    
    # Dummy macro features
    df['L_WTI_Ret'] = np.random.randn(len(df)) * 0.05  # Returns
    df['L_GPR'] = 150 + np.random.randn(len(df)) * 30  # GPR index
    
    # Regime features
    df['Regime_Label'] = (df['L_Regime'] > 0.5).astype(int)
    df['L_MS_Vol_Safe'] = df.groupby('Regime_Label')['Volatility']\
        .expanding().std().reset_index(level=0, drop=True).shift(1)
    df['L_MS_Vol_Safe'] = df['L_MS_Vol_Safe'].fillna(df['Volatility'].expanding().std().shift(1))
    
    # Interaction features
    df['L_State_S_Safe'] = df['L_Regime'] * df['Score_Centered']
    df['L_Crowd_Safe'] = df['Score_Centered'] * np.log1p(df['L_Inten'])
    
    return df


if __name__ == '__main__':
    # Test feature engineering
    print("Testing Feature Engineering...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Volatility': np.random.uniform(0.1, 0.5, 100),
        'Crisis_Prob': np.random.uniform(0, 1, 100),
        'Intensity': np.random.uniform(100, 300, 100),
        'Score': np.random.uniform(-0.5, 0.5, 100),
        'WTI': np.random.uniform(50, 100, 100),
        'gpr': np.random.uniform(100, 200, 100),
    })
    
    # Create features
    engineer = FeatureEngineer(train_cutoff='2023-01-01')
    df_features = engineer.create_all_features(sample_data)
    
    print(f"\nOriginal columns: {list(sample_data.columns)}")
    print(f"After engineering: {list(df_features.columns)}")
    print(f"New features: {set(df_features.columns) - set(sample_data.columns)}")
    
    # Get feature lists
    nprs1_features = engineer.get_nprs1_features(df_features)
    print(f"\nNPRS-1 features ({len(nprs1_features)}): {nprs1_features}")
    
    rf11_features = engineer.get_rf11_features(df_features)
    print(f"\nRF11 features ({len(rf11_features)}): {rf11_features}")
    
    # Validate
    is_valid = engineer.validate_features(df_features, rf11_features)
    print(f"\nFeatures valid: {is_valid}")
    
    # Summary
    summary = engineer.get_feature_summary(df_features)
    print(f"\nFeature summary:\n{summary[['mean', 'std', 'missing_pct']].head(10)}")
    
    print("\n✅ Feature engineering tests complete!")
