import pandas as pd

class RegimeDetector:
    def __init__(self, crisis_threshold=0.5, moderate_threshold=0.2):
        self.crisis_threshold = crisis_threshold
        self.moderate_threshold = moderate_threshold

    def classify_regime(self, probability):
        """Converts a probability float into a discrete regime string."""
        if probability >= self.crisis_threshold:
            return 'CRISIS'
        elif probability >= self.moderate_threshold:
            return 'MODERATE'
        else:
            return 'CALM'

    def get_regime_emoji(self, regime_str):
        mapping = {'CRISIS': '🔴', 'MODERATE': '🟡', 'CALM': '🟢'}
        return mapping.get(regime_str.upper(), '⚪')

    def detect_regime_shifts(self, df):
        """Identifies exactly when the market transitioned between regimes."""
        if 'Crisis_Prob' not in df.columns:
            return df
            
        df = df.copy()
        df['Regime_Label'] = df['Crisis_Prob'].apply(self.classify_regime)
        # Shift creates a column of the previous day's regime
        df['Previous_Regime'] = df['Regime_Label'].shift(1)
        # Identify rows where the regime changed
        df['Is_Shift'] = df['Regime_Label'] != df['Previous_Regime']
        
        return df
