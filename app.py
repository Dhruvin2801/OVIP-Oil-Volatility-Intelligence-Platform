import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Ensure the AI module is accessible
from modules.ai_engine import setup_rag_vector_db, get_ai_response

# ==========================================
# 1. CORE CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="OVIP // COMMAND", layout="wide", initial_sidebar_state="collapsed")

COLORS = {
    'bg': '#000000',           # Pitch Black
    'matrix': '#00FF41',       # Hacker Green
    'cyan': '#00F0FF',         # Radar Cyan
    'alert': '#FF003C',        # Threat Red
    'panel': 'rgba(0, 20, 0, 0.6)' # Translucent green glass
}

COUNTRIES = {
    'USA': {'lat': 37.09, 'lon': -95.71, 'flag': '🇺🇸', 'name': 'UNITED STATES'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'flag': '🇨🇳', 'name': 'PEOPLES REPUBLIC OF CHINA'},
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'flag': '🇮🇳', 'name': 'REPUBLIC OF INDIA'},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'flag': '🇦🇪', 'name': 'UNITED ARAB EMIRATES'},
    'SAUDI': {'lat': 23.89, 'lon': 45.08, 'flag': '🇸🇦', 'name': 'SAUDI ARABIA'}
}

# ==========================================
# 2. SCI-FI CSS INJECTION (NO SCROLLING)
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    /* Lock the screen, remove default padding */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg']};
        color: {COLORS['matrix']};
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important; 
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 98% !important;
    }}
    
    /* CRT Scanline Overlay */
    .stApp::after {{
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
                    linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 999;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }}

    /* Hide ugly scrollbars but allow internal scrolling */
    ::-webkit-scrollbar {{ display: none; }}

    /* Iron Man Cut-Corner Panels */
    .st-emotion-cache-1wivap2, div[data-testid="stVerticalBlock"] > div.element-container {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['matrix']};
        clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
        padding: 10px;
        box-shadow: inset 0 0 15px rgba(0, 255, 65, 0.1);
    }}

    /* Glowing Text */
    h1, h2, h3, h4, p, span {{
        font-family: 'Share Tech Mono', monospace !important;
        text-shadow: 0 0 8px {COLORS['matrix']};
        letter-spacing: 1.5px;
    }}
    
    /* Cyberpunk Buttons */
    .stButton>button {{
        background-color: transparent;
        color: {COLORS['cyan']};
        border: 1px solid {COLORS['cyan']};
        border-radius: 0px;
        width: 100%;
        text-transform: uppercase;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['cyan']};
        color: {COLORS['bg']};
        box-shadow: 0 0 15px {COLORS['cyan']};
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. STATE MANAGEMENT & DATA LOADING
# ==========================================
if 'target' not in st.session_state:
    st.session_state.target = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'intel'
if 'chat' not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "OVIP_DAEMON ONLINE. AWAITING TACTICAL QUERY..."}]

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('merged_final.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        # Fallback dummy data if CSV fails to load
        return pd.DataFrame({'Date': pd.date_range(start='1/1/2024', periods=100), 'WTI': 75.0, 'Volatility': 0.15, 'Crisis_Prob': 0.0})

df_main = load_data()
vec, tfidf, rag_df = setup_rag_vector_db(df_main)

# ==========================================
# 4. VIEW 1: THE 3D SATELLITE GLOBE
# ==========================================
def render_globe():
    st.markdown("<h2 style='text-align: center; color: #00F0FF;'>[ OVIP // GLOBAL_THREAT_MATRIX ]</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AWAITING SATELLITE UPLINK TARGET...</p>", unsafe_allow_html=True)
    
    # Globe Data
    df_nodes = pd.DataFrame.from_dict(COUNTRIES, orient='index').reset_index()
    df_nodes.columns = ['ID', 'lat', 'lon', 'flag', 'name']
    
    # WebGL Layers
    layer_nodes = pdk.Layer("ScatterplotLayer", df_nodes, get_position="[lon, lat]", get_color="[0, 255, 65, 200]", get_radius=400000, pickable=True)
    
    view_state = pdk.ViewState(latitude=25, longitude=40, zoom=1.2, pitch=45, bearing=15)
    
    st.pydeck_chart(pdk.Deck(layers=[layer_nodes], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10"), height=500)
    
    # Target Selection Selector
    st.markdown("```bash\n> INITIATE_NODE_LINK --target\n```")
    cols = st.columns(5)
    for i, (code, data) in enumerate(COUNTRIES.items()):
        with cols[i]:
            if st.button(f"{data['flag']} {code}"):
                st.session_state.target = code
                st.rerun()

# ==========================================
# 5. MAIN EVENT CHART (SAFE SUBPLOTS ENGINE)
# ==========================================
st.markdown("### > MACRO-EVENT VOLATILITY IMPACT MATRIX")

# Drop any blank dates to prevent rendering errors
chart_df = df_main.dropna(subset=['Date']).tail(150).copy()

# 1. Initialize Dual-Axis Chart safely using Plotly's native subplots engine
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 2. Trace 1: Volatility (Primary Y)
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility (Sigma)',
    line=dict(color="#00f0ff", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
), secondary_y=False)

# 3. Trace 2: WTI Price (Secondary Y)
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['WTI'], name='WTI Price ($)',
    line=dict(color="#00ff41", width=2, dash='dot')
), secondary_y=True)

# 4. Safe Event Markers 
try:
    max_vol = float(chart_df['Volatility'].max())
    mid_date = chart_df['Date'].iloc[len(chart_df)//3]
    late_date = chart_df['Date'].iloc[int(len(chart_df)//1.5)]

    fig.add_vline(x=mid_date, line_width=2, line_dash="dash", line_color="#FF003C")
    fig.add_vline(x=late_date, line_width=2, line_dash="dash", line_color="#FFD700")

    fig.add_annotation(x=mid_date, y=max_vol, text="[ US TARIFFS ]", showarrow=False, yshift=15, font=dict(color="#FF003C", family="Orbitron"))
    fig.add_annotation(x=late_date, y=max_vol * 0.9, text="[ OPEC POLICY ]", showarrow=False, yshift=15, font=dict(color="#FFD700", family="Orbitron"))
except Exception as e:
    pass # Silent failsafe

# 5. Safely update layout WITHOUT touching axes dictionaries
fig.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
    height=450, margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# 6. Update Axes using native methods (Immune to dictionary ValueErrors)
fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)')
fig.update_yaxes(title_text="Volatility (Sigma)", color="#00f0ff", showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)', secondary_y=False)
fig.update_yaxes(title_text="WTI Price (USD)", color="#00ff41", showgrid=False, secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
# ==========================================
# 6. APP ROUTER
# ==========================================
if st.session_state.target is None:
    render_globe()
else:
    render_dashboard()
