"""
OVIP - Helpers Utility
Provides formatting, UI consistency, and session state bootstrapping.
"""
import streamlit as st
import sys
from pathlib import Path

# Dynamically import the root config.py file
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def apply_custom_theme():
    """Wrapper to apply the Cyberpunk theme from the main config."""
    config.apply_custom_theme()

def initialize_session_state():
    """Bootstraps necessary session variables if they don't exist."""
    if 'initialized' not in st.session_state:
        st.session_state['selected_country'] = 'WTI'
        st.session_state['market_display'] = '🇺🇸 United States (WTI)'
        st.session_state['chat_history'] = []
        st.session_state['alerts'] = []
        st.session_state['initialized'] = True

def get_regime_info(prob: float) -> dict:
    """Returns the visual properties for a given crisis probability."""
    if prob > 0.5:
        return {'regime': 'CRISIS', 'emoji': '🔴', 'color': config.COLORS['danger']}
    elif prob > 0.1:
        return {'regime': 'MODERATE', 'emoji': '🟡', 'color': config.COLORS['warning']}
    return {'regime': 'CALM', 'emoji': '🟢', 'color': config.COLORS['accent_primary']}

def format_percentage(value: float) -> str:
    """Formats a decimal or float into a clean
