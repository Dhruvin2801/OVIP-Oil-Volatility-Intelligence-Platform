import streamlit as st

COLORS = {
    'background': '#0A192F', 'surface': '#112240',
    'accent_primary': '#64FFDA', 'accent_secondary': '#00FF41',
    'text_primary': '#E6F1FF', 'text_secondary': '#8892B0',
    'border': '#1E3A5F', 'danger': '#FF073A', 'warning': '#FFAA00'
}

def apply_custom_theme():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Rajdhani:wght@500;700&family=Roboto+Mono:wght@400;700&display=swap');
        .stApp {{ background-color: {COLORS['background']}; color: {COLORS['text_primary']}; font-family: 'Inter', sans-serif; }}
        h1, h2, h3, h4 {{ color: {COLORS['accent_primary']} !important; font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px; text-shadow: 0 0 10px rgba(100, 255, 218, 0.2); }}
        code, .stMetric, .stSelectbox {{ font-family: 'Roboto Mono', monospace !important; }}
        div[data-testid="stMetric"] {{ background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 5px; padding: 15px; border-left: 3px solid {COLORS['accent_primary']}; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        div[data-testid="stMetricValue"] {{ color: {COLORS['accent_primary']} !important; }}
    </style>
    """, unsafe_allow_html=True)
