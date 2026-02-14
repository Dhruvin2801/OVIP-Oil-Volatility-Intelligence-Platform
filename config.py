import streamlit as st

# 🟢 THE MISSING DICTIONARY: 
# Restored for the other pages, but updated with the Hacker Theme hex codes!
COLORS = {
    'background': '#020202',      # Deep Black
    'surface': '#050505',         # Terminal Black
    'accent_primary': '#00FF41',  # Matrix Green
    'accent_secondary': '#008F11',# Darker Green
    'text_primary': '#00FF41',    # Matrix Green
    'text_secondary': '#008F11',  # Darker Green
    'border': '#00FF41',          # Matrix Green
    'danger': '#FF003C',          # Blood Red
    'warning': '#FFD300'          # Cyber Yellow
}

def apply_custom_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        
        /* Deep Black Background & Terminal Font */
        .stApp {
            background-color: #020202;
            color: #00FF41;
            font-family: 'Share Tech Mono', monospace;
        }

        /* CRT Scanline Overlay */
        .stApp::after {
            content: " ";
            display: block;
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
            z-index: 2;
            background-size: 100% 2px, 3px 100%;
            pointer-events: none;
        }

        /* Hacker Glow Effects */
        h1, h2, h3, h4, p, span {
            font-family: 'Share Tech Mono', monospace !important;
            color: #00FF41 !important;
            text-shadow: 0px 0px 8px rgba(0, 255, 65, 0.6);
            letter-spacing: 1.5px;
        }

        /* Terminal Style Metric Boxes */
        div[data-testid="stMetric"] {
            background-color: #050505;
            border: 1px solid #00FF41;
            border-left: 5px solid #00FF41;
            padding: 15px;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
            position: relative;
        }
        
        div[data-testid="stMetric"]::before {
            content: "[SYS.METRIC]";
            font-size: 10px;
            color: #008F11;
            position: absolute;
            top: 2px;
            right: 5px;
        }

        /* Input Boxes & Buttons */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #000 !important;
            border: 1px solid #00FF41 !important;
            color: #00FF41 !important;
        }
        
        .stButton>button {
            background-color: #020202;
            color: #00FF41;
            border: 1px solid #00FF41;
            border-radius: 0px;
            font-family: 'Share Tech Mono', monospace;
            text-shadow: 0 0 5px #00FF41;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #00FF41;
            color: #000;
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.8);
        }

        /* Chat Terminal */
        .stChatInputContainer {
            border: 1px solid #00FF41 !important;
            background-color: #000 !important;
        }
    </style>
    """, unsafe_allow_html=True)
