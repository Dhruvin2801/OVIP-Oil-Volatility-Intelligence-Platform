import pandas as pd
import streamlit as st
from pathlib import Path

class DataLoader:
    def __init__(self):
        # Dynamically find the data folder
        self.data_dir = Path(__file__).resolve().parent.parent / 'data'
        
    @st.cache_data(ttl=3600)
    def merge_all_data(_self):
        try:
            # 1. Load the Single Master Dataset
            # (Make sure to rename your downloaded file to 'merged_final.csv' before putting it in the data folder)
            df_main = pd.read_csv(_self.data_dir / 'merged_final.csv')
            df_perf = pd.read_csv(_self.data_dir / 'data_model_performance_2025.csv')
            
            df_main['Date'] = pd.to_datetime(df_main['Date'])
            df_perf['Date'] = pd.to_datetime(df_perf['Date'])
            
            # 2. Merge Performance Data for the Dashboard Charts
            cols_to_use = ['Date', 'Predicted_Vol', 'Error', 'Uncertainty_Factor']
            df_combined = pd.merge(df_main, df_perf[cols_to_use], on='Date', how='left')
            
            return df_combined.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            st.error(f"Data Loader Error: {e}")
            return pd.DataFrame()

    def get_latest_metrics(self):
        df = self.merge_all_data()
        if df.empty: return None
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Determine Regime State dynamically
        regime_str = "CRISIS" if latest['Crisis_Prob'] > 0.5 else "MODERATE" if latest['Crisis_Prob'] > 0.1 else "CALM"
        
        return {
            'price': latest['WTI'],
            'price_change': ((latest['WTI'] - previous['WTI']) / previous['WTI']) * 100,
            'volatility': latest['Volatility'],
            'crisis_prob': latest['Crisis_Prob'],
            'regime': regime_str,
            'sentiment': latest.get('Score', 0)
        }

def get_data_loader():
    return DataLoader()
