import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Safely import AI engine
try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. SCI-FI HUD CONFIGURATION
# ==========================================
st.set_page_config(page_title="OVIP // TACTICAL_HUD", layout="wide", initial_sidebar_state="collapsed")

# Aggressive Sci-Fi Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* Full Viewport Lock */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000505 !important;
        color: #00ffcc !important;
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    /* Kill Streamlit Padding */
    .block-container {
        padding: 0.5rem 1rem !important;
        max-width: 100% !important;
    }
    header {visibility: hidden;}

    /* Sci-Fi Border & Glow */
    .stVerticalBlock > div.element-container {
        border: 1px solid rgba(0, 255, 204, 0.3);
        background: rgba(0, 20, 20, 0.4);
        padding: 10px;
        clip-path: polygon(0 10px, 10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%);
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.1);
        transition: 0.3s;
    }
    .stVerticalBlock > div.element-container:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffcc !important;
        text-shadow: 0 0 10px #00ffcc;
    }
    
    /* Custom Sci-Fi Buttons */
    .stButton>button {
        background: transparent !important;
        color: #00ffcc !important;
        border: 1px solid #00ffcc !important;
        border-radius: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
        clip-path: polygon(10% 0, 100% 0, 100% 70%, 90% 100%, 0 100%, 0 30%);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: rgba(0, 255, 204, 0.2) !important;
        box-shadow: 0 0 15px #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PIPELINE (LOCKED TO 'data/' FOLDER)
# ==========================================
@st.cache_data
def load_data():
    files = ['data/merged_final.csv', 'data/merged_final_corrected.csv', 'merged_final.csv']
    for f in files:
        if os.path.exists(f): 
            try:
                df = pd.read_csv(f)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            except: pass
    return pd.DataFrame({'Date': pd.date_range(end=pd.Timestamp.now(), periods=100), 'WTI': np.random.normal(75, 2, 100), 'Volatility': np.random.normal(0.15, 0.02, 100)})

df_main = load_data()

# ==========================================
# 3. HUD STATE & COUNTRY DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'STABLE', 'intel': 'Refinery intake optimized. Strategic reserve utilization: 82%.', 'color': '#00ffcc'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'CAUTION', 'intel': 'Permian basin output plateauing. SPR refill logic active.', 'color': '#00ccff'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'ELEVATED', 'intel': 'Demand cooling detected. Import quota reallocation in progress.', 'color': '#ff3333'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'intel': 'Dark fleet tracking active. Urals discount compression.', 'color': '#ff0000'},
    'SAUDI': {'lat': 23.89, 'lon': 45.08, 'risk': 'CAUTION', 'intel': 'OPEC+ compliance strict. Spare capacity: 2.8M bpd.', 'color': '#ffcc00'}
}

# ==========================================
# 4. VIEW: TACTICAL GLOBE (HOME)
# ==========================================
if st.session_state.target is None:
    st.markdown("<h1 style='text-align:center;'>[ OVIP // GLOBAL_THREAT_MATRIX ]</h1>", unsafe_allow_html=True)
    
    lats, lons, names = [], [], []
    for k, v in COUNTRIES.items():
        names.append(k); lats.append(v['lat']); lons.append(v['lon'])
    
    # Sci-Fi Holographic Globe
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=12, color='#00ffcc', symbol='square', line=dict(width=1, color='#fff')),
        textfont = dict(family="Share Tech Mono", size=14, color="#00ffcc"),
        textposition = "top center"
    ))
    
    fig_globe.update_geos(
        projection_type="orthographic", showcoastlines=True, coastlinecolor="rgba(0, 255, 204, 0.3)",
        showland=True, landcolor="#001a1a", showocean=True, oceancolor="#000505",
        showcountries=True, countrycolor="rgba(0, 255, 204, 0.2)",
        center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
    )
    fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    # Render Globe
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
        st.rerun()

    # Matrix Node Selector
    st.markdown("<p style='text-align:center;'>SELECT_NODE_FOR_UPLINK:</p>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, n in enumerate(names):
        if cols[i].button(n): st.session_state.target = n; st.rerun()

# ==========================================
# 5. VIEW: DATA UPLINK (DASHBOARD)
# ==========================================
else:
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    
    # HUD Header
    col_t, col_b = st.columns([5, 1])
    col_t.markdown(f"<h2 style='color:{intel['color']}'>ACCESSING_NODE::{target}</h2>", unsafe_allow_html=True)
    if col_b.button("DISCONNECT"): st.session_state.target = None; st.rerun()
    
    st.markdown("---")
    
    # High-Density Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WTI_CRUDE", f"${latest.get('WTI', 0):.2f}")
    m2.metric("VOL_SIGMA", f"{latest.get('Volatility', 0):.3f}")
    m3.metric("THREAT_LVL", intel['risk'])
    m4.metric("NODE_STATUS", "ACTIVE")
    
    st.markdown("---")
    
    # Data Visualization + Intel Log
    c_chart, c_log = st.columns([2.5, 1])
    
    with c_chart:
        chart_df = df_main.tail(100)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Volatility'], line=dict(color=intel['color'], width=3), fill='tozeroy'), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], line=dict(color="#ffffff", width=1, dash='dot')), secondary_y=True)
        fig.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,20,0.2)')
        st.plotly_chart(fig, use_container_width=True)
        
    with c_log:
        st.markdown(f"<div style='border:1px solid {intel['color']}; padding:15px; height:400px;'>", unsafe_allow_html=True)
        st.markdown(f"### > {target}_INTEL_LOG")
        st.write(intel['intel'])
        st.markdown("---")
        # Floating AI Button
        if st.button("💬 INIT_DAEMON_AI"):
            st.info("Daemon UPlink initiated. Type command in sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for AI Chat (Always accessible)
with st.sidebar:
    st.title("OVIP_DAEMON_AI")
    if prompt := st.chat_input("Enter command..."):
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"): st.write("Processing tactical query...")
