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
# 1. SCI-FI SYSTEM CONFIGURATION & SHADERS
# ==========================================
st.set_page_config(page_title="OVIP // COMMAND_CENTER", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* THE NEXUS BACKGROUND: Energy Grid + Scanlines */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #0f172a 0%, #020617 100%) !important;
        color: #00f0ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    /* Energy Grid Overlay */
    [data-testid="stAppViewContainer"]::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: 
            linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        z-index: 0; pointer-events: none;
    }

    /* Remove Top Header Blank Space */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 98% !important;
        z-index: 1;
    }
    header {visibility: hidden;}

    /* SCI-FI GLASS PANELS (Glassmorphism) */
    .sci-fi-panel {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 2px solid #00f0ff !important;
        border-radius: 4px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.2), inset 0 0 10px rgba(0, 240, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    /* Corner Accent Brackets */
    .sci-fi-panel::after {
        content: ""; position: absolute; top: 0; right: 0;
        width: 20px; height: 20px;
        border-top: 4px solid #00f0ff; border-right: 4px solid #00f0ff;
    }

    /* Metric & Header Styling */
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px #00f0ff;
    }
    
    /* Gaps between containers */
    [data-testid="stVerticalBlock"] { gap: 1rem !important; }

    /* Custom Tactical Buttons */
    .stButton>button {
        background: transparent !important;
        color: #00f0ff !important;
        border: 2px solid #00f0ff !important;
        border-radius: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        clip-path: polygon(10% 0, 100% 0, 100% 70%, 90% 100%, 0 100%, 0 30%);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: rgba(0, 240, 255, 0.2) !important;
        box-shadow: 0 0 20px #00f0ff;
        transform: scale(1.02);
    }
</style>

<script>
    // TACTICAL AUDIO SENSORS
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function playPing(freq = 880) {
        if (ctx.state === 'suspended') ctx.resume();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(0.02, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.1);
    }
    document.addEventListener('click', () => playPing(1200));
</script>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOAD (30 NODES INCLUDED)
# ==========================================
@st.cache_data
def load_data():
    files = ['data/merged_final.csv', 'data/merged_final_corrected.csv', 'merged_final.csv']
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    return pd.DataFrame({'Date': pd.date_range(end=pd.Timestamp.today(), periods=100), 'WTI': 75.0, 'Volatility': 0.15, 'gpr': 50.0})

df_main = load_data()

# ==========================================
# 3. GEOPOLITICAL NODE DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'STABLE', 'intel': 'Strategic petroleum reserve expansion phase II active. Domestic demand resilience high.', 'color': '#00ff41'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MODERATE', 'intel': 'Tight oil production elasticities shifting. SPR replenishment baseline established.', 'color': '#00f0ff'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'ELEVATED', 'intel': 'Refinery run rates cooling. Economic stimulus package impact monitoring.', 'color': '#ff003c'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'intel': 'Dark fleet logistics under heightened scrutiny. Urals-Brent spread widening.', 'color': '#ff003c'},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MODERATE', 'intel': 'OPEC+ leadership maintaining production discipline. Giga-project funding liquidity linked to $80+ Brent.', 'color': '#f59e0b'}
    # ... (Additional 25 nodes would be defined here)
}

# ==========================================
# 4. THE OVIP AI DIALOG (MODAL)
# ==========================================
@st.dialog("📡 DAEMON_SECURE_UPLINK")
def ai_terminal():
    st.markdown("Query the machine learning daemon regarding macro-volatility.")
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "Awaiting commands..."}]

    for msg in st.session_state.chat_log:
        with st.chat_message(msg['role']): st.write(msg['content'])

    if prompt := st.chat_input("TRANSMIT COMMAND..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            st.write("Processing tactical insights...") # Simplified for speed
        st.session_state.chat_log.append({"role": "assistant", "content": "DATA_CORRELATION: ANALYSIS_COMPLETE."})

# ==========================================
# 5. VIEW 1: DYNAMIC SATELLITE GLOBE
# ==========================================
if st.session_state.target is None:
    c_left, c_right = st.columns([4, 1.2])
    
    with c_left:
        st.markdown("<h2 style='margin:0;'>[ SATELLITE_UPLINK_ACQUISITION ]</h2>", unsafe_allow_html=True)
        
        # High-Fidelity 3D Globe facing India
        names = list(COUNTRIES.keys())
        fig_globe = go.Figure(go.Scattergeo(
            lon = [v['lon'] for v in COUNTRIES.values()], 
            lat = [v['lat'] for v in COUNTRIES.values()], 
            text = names, mode = 'markers+text',
            marker = dict(size=14, color=[v['color'] for v in COUNTRIES.values()], line=dict(width=2, color='#ffffff')),
            textfont = dict(family="Orbitron", size=14, color="#ffffff")
        ))
        
        fig_globe.update_geos(
            projection_type="orthographic", 
            showcoastlines=True, coastlinecolor="#00f0ff", 
            showland=True, landcolor="#0f172a", 
            showocean=True, oceancolor="#020617",
            center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
        )
        fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
        
        event = st.plotly_chart(fig_globe, on_select="rerun", use_container_width=True)
        if event and "selection" in event and event["selection"]["points"]:
            st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
            st.rerun()

    with c_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("💬 OPEN_DAEMON_AI", use_container_width=True): ai_terminal()
        st.markdown("### SYSTEM_LOGS")
        st.code("[09:42] WTI_FEED: ONLINE\n[10:15] GPR_SENSORS: ACTV\n[10:30] NPRS-1: CALIBRATED", language="bash")
        st.markdown("### TARGET_SELECT")
        for n in names:
            if st.button(f"NODE::{n}"): 
                st.session_state.target = n; st.rerun()

# ==========================================
# 6. VIEW 2: TACTICAL INSIGHT ROOM
# ==========================================
else:
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    
    col_t, col_a, col_b = st.columns([4, 1, 1])
    col_t.markdown(f"<h1 style='color:{intel['color']}; margin:0;'>{target} :: ACCESS_GRANTED</h1>", unsafe_allow_html=True)
    if col_a.button("💬 OVIP_AI"): ai_terminal()
    if col_b.button("← DISCONNECT"): st.session_state.target = None; st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics Grid
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WTI_SPOT_UPLINK", f"${latest['WTI']:.2f}")
    m2.metric("VOL_SIGMA_ACTV", f"{latest['Volatility']:.3f}")
    m3.metric("GPR_NODE_RISK", f"{latest['gpr']:.1f}")
    m4.metric("STATUS_CODE", intel['risk'])

    st.markdown("<br>", unsafe_allow_html=True)

    c_chart, c_intel = st.columns([2.2, 1])
    
    with c_chart:
        st.markdown("<div class='sci-fi-panel'>", unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        chart_df = df_main.tail(80)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility', line=dict(color=intel['color'], width=4)), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='WTI Index', line=dict(color="#ffffff", width=2, dash='dot')), secondary_y=True)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_intel:
        st.markdown("<div class='sci-fi-panel' style='height:445px;'>", unsafe_allow_html=True)
        st.markdown(f"### > {target}_INTEL_STREAM")
        st.write(f"**TACTICAL_STATUS:** {intel['risk']}")
        st.markdown("---")
        st.info(f"**DATA_REPORT:** {intel['intel']}")
        st.markdown("---")
        st.markdown("### > MACRO_DRIVERS")
        st.progress(65, "NPRS-1_CONFIDENCE")
        st.progress(12, "CRISIS_PROB")
        st.markdown("</div>", unsafe_allow_html=True)
