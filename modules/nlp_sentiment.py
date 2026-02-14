import pandas as pd
import numpy as np

class SentimentAnalyzer:
    @staticmethod
    def calculate_sentiment_momentum(df, window=3):
        """
        Calculates the velocity of sentiment changes.
        A sharp drop in sentiment momentum often precedes a volatility spike.
        """
        if 'Score' not in df.columns or 'Intensity' not in df.columns:
            return df
            
        df = df.copy()
        # Create a weighted sentiment metric
        df['Weighted_Sentiment'] = df['Score'] * np.log1p(df['Intensity'])
        
        # Calculate momentum (Rate of Change)
        df['Sentiment_MA'] = df['Weighted_Sentiment'].rolling(window=window).mean()
        df['Sentiment_Momentum'] = df['Sentiment_MA'].diff()
        
        return df
        
    @staticmethod
    def get_sentiment_alert_level(current_score, current_intensity):
        """Categorizes current sentiment for the Threat Matrix."""
        if current_score < -0.10 and current_intensity > 200:
            return {'level': 8.9, 'status': 'CRITICAL', 'description': 'High Volume Negative News'}
        elif current_score < -0.05:
            return {'level': 6.5, 'status': 'WARNING', 'description': 'Elevated Negative Sentiment'}
        else:
            return {'level': 2.5, 'status': 'NORMAL', 'description': 'Baseline News Flow'}
