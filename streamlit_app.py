import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from modules.ai_engine import setup_rag_vector_db, get_ai_response
from config import COLORS, COUNTRIES

# 1. INITIALIZE COMMAND CENTER
st.set_page_config(page_title="OVIP // TACTICAL", layout="wide", initial_sidebar_state="collapsed")

# 2. THE "JARVIS" UI ENGINE (HTML/JS/CSS INJECTION)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    :root {{
        --matrix: #00FF41;
        --cyan: #00F0FF;
        --bg: #000500;
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg) !important;
        color: var(--matrix) !important;
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important;
    }}

    /* SCREEN FLICKER & SCANLINES (The "Hacker" Texture) */
    .stApp {{
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), 
                    linear-gradient(90deg, rgba(255, 0, 0, 0.03), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.03));
        background-size: 100% 3px, 3px 100%;
    }}

    /* ANGLED HUD PANELS */
    .element-container, div[data-testid="column"] {{
        background: rgba(0, 255, 65, 0.03) !important;
        border: 1px solid rgba(0, 255, 65, 0.3) !important;
        clip-path: polygon(0 15px, 15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
        padding: 20px !important;
        transition: all 0.3s ease;
    }}
    .element-container:hover {{
        border-color: var(--cyan) !important;
        background: rgba(0, 240, 255, 0.05) !important;
    }}

    /* TAB BUTTONS (Mission Impossible Style) */
    .stButton>button {{
        background: transparent !important;
        color: var(--matrix) !important;
        border: 1px solid var(--matrix) !important;
        border-radius: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 3px !important;
        text-transform: uppercase;
        clip-path: polygon(10% 0, 100% 0, 100% 70%, 90% 100%, 0 100%, 0 30%);
    }}
    </style>
    
    <script>
    // TACTICAL SOUND ENGINE
    const playBeep = () => {{
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        oscillator.type = 'square';
        oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
        gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime);
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.05);
    }};
    
    document.addEventListener('click', () => playBeep());
    </script>
""", unsafe_allow_html=True)

# 3. CORE LOGIC (SESSION STATES)
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

# 4. GLOBE INTERFACE (SATELLITE VIEW)
def render_globe():
    st.markdown("<h1 style='text-align: center; font-family: Orbitron;'>OVIP // SATELLITE_LINK</h1>", unsafe_allow_html=True)
    
    # Custom 3D Map using Deck.GL Arcs (The "Kaspersky" Look)
    arc_data = pd.DataFrame([
        {'s_lon': -95, 's_lat': 37, 't_lon': 53, 't_lat': 23}, # USA to UAE
        {'s_lon': 104, 's_lat': 35, 't_lon': 78, 't_lat': 20}, # China to India
    ])
    
    layer = pdk.Layer(
        "ArcLayer", arc_data,
        get_source_position='[s_lon, s_lat]', get_target_position='[t_lon, t_lat]',
        get_source_color='[0, 255, 65, 100]', get_target_color='[0, 240, 255, 100]',
        width=2
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer], 
        initial_view_state=pdk.ViewState(latitude=20, longitude=50, zoom=1, pitch=40),
        map_style="mapbox://styles/mapbox/dark-v10"
    ), height=500)

    # SELECT TARGET
    st.markdown("### > SELECT_NODE_FOR_UPLINK")
    cols = st.columns(len(COUNTRIES))
    for i, (code, info) in enumerate(COUNTRIES.items()):
        if cols[i].button(f"{info['flag']} {code}"):
            st.session_state.target = code
            st.rerun()

# 5. DASHBOARD INTERFACE (THE COMMAND CENTER)
def render_dashboard():
    target = st.session_state.target
    st.markdown(f"<h2 style='font-family: Orbitron; color: #00F0FF;'>[ ACCESS_NODE_0{list(COUNTRIES.keys()).index(target)+1} :: {target} ]</h2>", unsafe_allow_html=True)
    
    c_nav, c_main, c_side = st.columns([0.8, 3, 1.2])
    
    with c_nav:
        st.markdown("#### SYSTEM")
        if st.button("DISCONNECT"): st.session_state.target = None; st.rerun()
        st.markdown("#### MODULES")
        if st.button("INTEL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("DAEMON"): st.session_state.active_tab = 'ai'; st.rerun()

    with c_main:
        if st.session_state.active_tab == 'intel':
            st.markdown("#### > TACTICAL_PRICE_FEED")
            # YOUR VOLATILITY PLOT HERE
            st.info("DATA_STREAM_PENDING...")
        else:
            # YOUR AI CHAT HERE
            st.success("DAEMON_V3_ONLINE")

    with c_side:
        st.markdown("#### > LOG_REPORTS")
        st.markdown("```bash\n[08:42] OPEC_ALERT: HIGH\n[09:15] NPRS-1: UP_TREND\n[10:02] SUEZ_TRANSIT: NORMAL\n```")
        st.markdown("#### > SYSTEM_RESOURCES")
        st.progress(69, "NPRS-1 ACCURACY")
        st.progress(12, "CRISIS_PROBABILITY")

# ROUTER
if st.session_state.target is None:
    render_globe()
else:
    render_dashboard()
