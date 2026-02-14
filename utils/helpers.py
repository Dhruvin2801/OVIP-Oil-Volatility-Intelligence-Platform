"""
OVIP - Helper Utilities
Common utility functions used across the application
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import io
from typing import Any, Dict, Optional

# --- SAFE IMPORTS WITH FALLBACKS ---
# We try to import from config, but provide fallbacks so the app NEVER crashes
try:
    from config import COLORS
except ImportError:
    COLORS = {
        'background': '#0A192F', 'surface': '#112240',
        'accent_primary': '#64FFDA', 'accent_secondary': '#00FF41',
        'text_primary': '#E6F1FF', 'text_secondary': '#8892B0',
        'border': '#1E3A5F', 'danger': '#FF073A', 'warning': '#FFAA00'
    }

# Fallback dictionaries for Regime styling and Session State
REGIME_COLORS = {'CALM': '#64FFDA', 'MODERATE': '#FFAA00', 'CRISIS': '#FF073A'}
REGIME_EMOJIS = {'CALM': '🟢', 'MODERATE': '🟡', 'CRISIS': '🔴'}

DEFAULT_SESSION_STATE = {
    'selected_market': 'WTI',
    'market_display': '🇺🇸 United States (WTI)',
    'chat_history': []
}


def apply_custom_theme():
    """Apply custom CSS styling to Streamlit app"""
    
    st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: {COLORS['background']};
        color: {COLORS['text_primary']};
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['accent_primary']} !important;
        font-family: 'Rajdhani', sans-serif !important;
    }}
    
    /* Paragraphs */
    p, li, span {{
        color: {COLORS['text_primary']} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {COLORS['surface']};
        color: {COLORS['accent_primary']};
        border: 1px solid {COLORS['accent_primary']};
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-family: 'Roboto Mono', monospace;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['accent_primary']};
        color: {COLORS['background']};
        box-shadow: 0 0 20px {COLORS['accent_primary']};
    }}
    
    /* Primary button */
    .stButton>button[kind="primary"] {{
        background-color: {COLORS['accent_primary']};
        color: {COLORS['background']};
        border: 1px solid {COLORS['accent_primary']};
    }}
    
    /* Metrics */
    div[data-testid="stMetricValue"] {{
        color: {COLORS['accent_primary']} !important;
        font-size: 2rem !important;
    }}
    
    div[data-testid="stMetricLabel"] {{
        color: {COLORS['text_secondary']} !important;
    }}
    
    /* Selectbox */
    .stSelectbox>div>div {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
    }}
    
    /* Text input */
    .stTextInput>div>div>input {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 2px solid {COLORS['border']};
    }}
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {{
        background-color: {COLORS['surface']};
        border-left: 3px solid {COLORS['accent_primary']};
        color: {COLORS['text_primary']};
    }}
    
    /* Horizontal rule */
    hr {{
        border: 1px solid {COLORS['accent_primary']};
    }}
    
    /* Custom cards */
    .ovip-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state with defaults if not already set"""
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_regime_info(crisis_prob: float) -> Dict[str, Any]:
    """Get regime information from crisis probability"""
    if crisis_prob < 0.3:
        regime = 'CALM'
    elif crisis_prob < 0.7:
        regime = 'MODERATE'
    else:
        regime = 'CRISIS'
    
    return {
        'regime': regime,
        'color': REGIME_COLORS[regime],
        'emoji': REGIME_EMOJIS[regime],
        'probability': crisis_prob,
    }


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"


def calculate_change(current: float, previous: float) -> Dict[str, Any]:
    """Calculate absolute and percentage change"""
    if previous == 0:
        return {'absolute_change': 0, 'pct_change': 0, 'direction': 'neutral'}
    
    abs_change = current - previous
    pct_change = (abs_change / previous) * 100
    direction = 'up' if abs_change > 0 else 'down' if abs_change < 0 else 'neutral'
    
    return {
        'absolute_change': abs_change,
        'pct_change': pct_change,
        'direction': direction,
    }


def create_downloadable_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to downloadable CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')


def create_downloadable_excel(df: pd.DataFrame) -> bytes:
    """
    FIXED: Convert DataFrame to downloadable Excel safely.
    Uses pd.ExcelWriter to ensure openpyxl saves correctly to the Bytes buffer.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='OVIP Data')
    return buffer.getvalue()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is 0"""
    try:
        return numerator / denominator if denominator != 0 else default
    except ZeroDivisionError:
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_date_range(period: str) -> tuple:
    """Get date range for common periods"""
    end_date = datetime.now()
    
    if period == '1d':
        start_date = end_date - timedelta(days=1)
    elif period == '7d':
        start_date = end_date - timedelta(days=7)
    elif period == '30d':
        start_date = end_date - timedelta(days=30)
    elif period == '90d':
        start_date = end_date - timedelta(days=90)
    elif period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == 'ytd':
        start_date = datetime(end_date.year, 1, 1)
    else:  # 'all'
        start_date = datetime(2002, 1, 1)
    
    return start_date, end_date


def create_alert_box(title: str, message: str, alert_type: str = 'info'):
    """Create styled alert box"""
    color_map = {
        'info': COLORS['accent_primary'],
        'success': COLORS['accent_secondary'],
        'warning': COLORS['warning'],
        'error': COLORS['danger'],
    }
    
    color = color_map.get(alert_type, COLORS['accent_primary'])
    
    st.markdown(f"""
    <div style="background-color: {COLORS['surface']}; border-left: 4px solid {color}; 
                padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
        <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{title}</h4>
        <p style="margin: 0; color: {COLORS['text_primary']};">{message}</p>
    </div>
    """, unsafe_allow_html=True)
