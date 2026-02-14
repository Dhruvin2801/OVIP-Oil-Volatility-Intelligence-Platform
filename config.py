"""
OVIP - Oil Volatility Intelligence Platform
Configuration and Constants
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = DATA_DIR / 'models'
ASSETS_DIR = BASE_DIR / 'assets'
PAGES_DIR = BASE_DIR / 'pages'

# Data files
HISTORICAL_DATA_PATH = DATA_DIR / 'merged_final.csv'
REGIME_PROBS_PATH = DATA_DIR / 'HONEST_REGIME_PROBS.csv'
MODEL_PERFORMANCE_PATH = DATA_DIR / 'data_model_performance_2025.csv'

# Model files
NPRS1_MODEL_PATH = MODELS_DIR / 'nprs1_classifier.pkl'
RF11_MODEL_PATH = MODELS_DIR / 'rf11_regressor.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'

# ============================================================================
# THEME & STYLING
# ============================================================================
COLORS = {
    'background': '#0A192F',      # Deep navy blue (main background)
    'surface': '#112240',         # Slightly lighter navy (cards/panels)
    'accent_primary': '#64FFDA',  # Matrix green (highlights, success)
    'accent_secondary': '#00FF41', # Bright green (live data)
    'text_primary': '#E6F1FF',    # Off-white (main text)
    'text_secondary': '#8892B0',  # Muted blue-gray (secondary text)
    'warning': '#FFAA00',         # Amber (warnings)
    'danger': '#FF073A',          # Red (alerts, crisis)
    'border': '#1E3A5F',          # Dark blue (borders, dividers)
    'chart_line': '#64FFDA',      # Green (chart lines)
    'chart_area': 'rgba(100, 255, 218, 0.1)',  # Transparent green
}

FONTS = {
    'main': 'Roboto Mono',
    'headings': 'Rajdhani',
    'body': 'Inter',
}

# ============================================================================
# COUNTRY CONFIGURATIONS
# ============================================================================
COUNTRIES = {
    'WTI': {
        'name': 'United States',
        'flag': '🇺🇸',
        'oil_type': 'WTI Crude',
        'full_name': 'West Texas Intermediate',
        'lat': 37.0902,
        'lon': -95.7129,
        'market_code': 'NYMEX:CL',
    },
    'BRENT': {
        'name': 'United Kingdom',
        'flag': '🇬🇧',
        'oil_type': 'Brent Crude',
        'full_name': 'Brent Crude Oil',
        'lat': 55.3781,
        'lon': -3.4360,
        'market_code': 'ICE:BRN',
    },
    'DUBAI': {
        'name': 'UAE',
        'flag': '🇦🇪',
        'oil_type': 'Dubai Crude',
        'full_name': 'Dubai Crude Oil',
        'lat': 23.4241,
        'lon': 53.8478,
        'market_code': 'DUBAI',
    },
    'ARAB_LIGHT': {
        'name': 'Saudi Arabia',
        'flag': '🇸🇦',
        'oil_type': 'Arab Light',
        'full_name': 'Saudi Arabian Arab Light',
        'lat': 23.8859,
        'lon': 45.0792,
        'market_code': 'ARAMCO:LIGHT',
    },
    'URALS': {
        'name': 'Russia',
        'flag': '🇷🇺',
        'oil_type': 'Urals',
        'full_name': 'Russian Urals Crude',
        'lat': 61.5240,
        'lon': 105.3188,
        'market_code': 'URALS',
    },
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# NPRS-1 Binary Classifier Config
NPRS1_CONFIG = {
    'name': 'NPRS-1 Binary Direction Classifier',
    'features': ['L_Vol', 'L_Regime', 'L_Inten', 'L_GPR', 'L_Accel'],
    'n_estimators': 300,
    'max_depth': 5,
    'random_state': 42,
    'accuracy': 0.682,  # 68.2%
}

# 11-Pillar Regression Config
RF11_CONFIG = {
    'name': '11-Pillar Volatility Level Predictor',
    'features': [
        'L_Vol', 'L_Regime', 'L_Inten', 'L_WTI_Ret', 'L_GPR',
        'L_Accel', 'L_News_Shk', 'L_MS_Vol_Safe',
        'L_State_S_Safe', 'L_Crowd_Safe', 'L_Vol_Std'
    ],
    'n_estimators': 500,
    'max_depth': 7,
    'random_state': 42,
    'r2_score': 0.225,  # 22.5%
}

# Feature importance (from actual model)
FEATURE_IMPORTANCE = {
    'L_Regime': 0.502,
    'L_Vol': 0.119,
    'L_Inten': 0.092,
    'L_Accel': 0.042,
    'L_GPR': 0.013,
    'L_News_Shk': 0.009,
    'L_WTI_Ret': 0.007,
    'L_Vol_Std': 0.003,
    'L_MS_Vol_Safe': 0.000,
    'L_Crowd_Safe': 0.000,
    'L_State_S_Safe': -0.002,
}

# ============================================================================
# REGIME THRESHOLDS
# ============================================================================
REGIME_THRESHOLDS = {
    'CALM': (0.0, 0.3),      # Crisis probability < 0.3
    'MODERATE': (0.3, 0.7),  # Crisis probability 0.3-0.7
    'CRISIS': (0.7, 1.0),    # Crisis probability > 0.7
}

REGIME_COLORS = {
    'CALM': '#64FFDA',       # Green
    'MODERATE': '#FFAA00',   # Amber
    'CRISIS': '#FF073A',     # Red
}

REGIME_EMOJIS = {
    'CALM': '🟢',
    'MODERATE': '🟡',
    'CRISIS': '🔴',
}

# ============================================================================
# ALERT CONFIGURATIONS
# ============================================================================
ALERT_TYPES = {
    'REGIME_SHIFT': 'Regime State Change',
    'PRICE_SPIKE': 'Price Movement Alert',
    'VOL_SPIKE': 'Volatility Spike',
    'MODEL_SIGNAL': 'High Confidence Signal',
    'CUSTOM': 'Custom Alert',
}

NOTIFICATION_METHODS = ['Email', 'SMS', 'In-App', 'Webhook']

# ============================================================================
# API CONFIGURATIONS
# ============================================================================
OPENAI_MODEL = 'gpt-4-turbo-preview'
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 1000

# ============================================================================
# DISPLAY CONFIGURATIONS
# ============================================================================
CHART_HEIGHT = 400
CHART_WIDTH = 1000

# Number of data points to display
DEFAULT_HISTORY_MONTHS = 60  # 5 years
FORECAST_HORIZON_DAYS = 90   # 3 months

# Performance metrics display periods
PERF_PERIODS = {
    '7d': 7,
    '30d': 30,
    'all': 'all',
}

# ============================================================================
# TIME CONFIGURATIONS
# ============================================================================
TIMEZONE = 'UTC'
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Data refresh intervals (in seconds)
REFRESH_INTERVALS = {
    'live_data': 60,      # 1 minute
    'model_update': 3600, # 1 hour
    'alerts': 900,        # 15 minutes
}

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================
MAX_VOLATILITY = 2.0  # Maximum reasonable volatility
MIN_VOLATILITY = 0.0  # Minimum volatility
MAX_PRICE = 300.0     # Maximum reasonable oil price ($/barrel)
MIN_PRICE = 10.0      # Minimum reasonable oil price

# ============================================================================
# EXPORT CONFIGURATIONS
# ============================================================================
EXPORT_FORMATS = {
    'CSV': 'text/csv',
    'EXCEL': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'PDF': 'application/pdf',
    'JSON': 'application/json',
}

# ============================================================================
# REPORT TEMPLATES
# ============================================================================
REPORT_TYPES = [
    'Weekly Executive Summary',
    'Monthly Performance Review',
    'Quarterly Strategic Forecast',
    'Annual Analysis',
    'Custom Report',
]

# ============================================================================
# FEATURE CATEGORIES
# ============================================================================
FEATURE_CATEGORIES = {
    'Persistence': ['L_Vol', 'L_Vol_Std', 'L_Accel'],
    'Regime': ['L_Regime', 'L_MS_Vol_Safe', 'L_State_S_Safe'],
    'NLP': ['L_Inten', 'L_Crowd_Safe', 'L_News_Shk'],
    'Macro': ['L_WTI_Ret', 'L_GPR'],
}

# ============================================================================
# LOGGING CONFIGURATIONS
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# SESSION STATE DEFAULTS
# ============================================================================
DEFAULT_SESSION_STATE = {
    'selected_country': 'WTI',
    'current_page': 'dashboard',
    'chat_history': [],
    'alerts': [],
    'user_preferences': {
        'theme': 'dark',
        'notifications_enabled': True,
        'auto_refresh': True,
    },
}

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable with fallback"""
    return os.getenv(key, default)

# API Keys (from .env file)
OPENAI_API_KEY = get_env_var('OPENAI_API_KEY', '')
TWILIO_ACCOUNT_SID = get_env_var('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = get_env_var('TWILIO_AUTH_TOKEN', '')
EMAIL_USERNAME = get_env_var('EMAIL_USERNAME', '')
EMAIL_PASSWORD = get_env_var('EMAIL_PASSWORD', '')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_regime_from_prob(crisis_prob: float) -> str:
    """Determine regime from crisis probability"""
    if crisis_prob < REGIME_THRESHOLDS['CALM'][1]:
        return 'CALM'
    elif crisis_prob < REGIME_THRESHOLDS['MODERATE'][1]:
        return 'MODERATE'
    else:
        return 'CRISIS'

def get_regime_color(regime: str) -> str:
    """Get color for regime"""
    return REGIME_COLORS.get(regime, COLORS['text_secondary'])

def get_regime_emoji(regime: str) -> str:
    """Get emoji for regime"""
    return REGIME_EMOJIS.get(regime, '⚪')

def validate_volatility(vol: float) -> bool:
    """Check if volatility is within reasonable bounds"""
    return MIN_VOLATILITY <= vol <= MAX_VOLATILITY

def validate_price(price: float) -> bool:
    """Check if price is within reasonable bounds"""
    return MIN_PRICE <= price <= MAX_PRICE
