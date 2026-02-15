import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import os
import time

# 1. INITIALIZE SYSTEM
st.set_page_config(page_title="OVIP // TACTICAL_OS", layout="wide", initial_sidebar_state="collapsed")

# 2. THE CINEMATIC SHADER ENGINE (CSS/JS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');

    /* BASE THEME */
    :root {
        --matrix-green: #00FF41;
        --cyan-glow: #00F0FF;
        --bg-dark: #000800;
        --panel-bg: rgba(0, 20, 0, 0.7);
    }

    /* REMOVE ALL STREAMLIT DEFAULTS */
    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: var(--bg-dark) !important;
        color: var(--matrix-green) !important;
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important;
        padding: 0 !important;
    }

    /* THE "ARWES" HEX GRID BACKGROUND */
    [data-testid="stAppViewContainer"]::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: 
            linear-gradient(rgba(0, 255, 65, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 65, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none; z-index: 0;
    }

    /* CRT SCANLINES & FLICKER */
    .stApp::after {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0.1) 1px, transparent 1px, transparent 2px);
        pointer-events: none; z-index: 1000; opacity: 0.3;
    }

    /* THE TACTICAL HUD PANELS */
    div.stVerticalBlock > div.element-container {
        background: var(--panel-bg);
        border: 1px solid var(--matrix-green);
        position: relative;
        margin: 10px;
        /* Corner Braces */
        clip-path: polygon(
            0 20px, 20px 0, 
            calc(100% - 20px) 0, 100% 20px, 
            100% calc(100% - 20px), calc(100% - 20px) 100%, 
            20px 100%, 0 calc(100% - 20px)
        );
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.2), 0 0 15px rgba(0, 255, 65, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div.stVerticalBlock > div.element-container:hover {
        border-color: var(--cyan-glow);
        box-shadow: inset 0 0 20px rgba(0, 240, 255, 0.2), 0 0 15px rgba(0, 240, 255, 0.2);
    }

    /* CUSTOM HUD BUTTONS */
    .stButton > button {
        background: transparent !important;
        color: var(--matrix-green) !important;
        border: 1px solid var(--matrix-green) !important;
        border-radius: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        background: rgba(0, 255, 65, 0.1) !important;
        box-shadow: 0 0 20px var(--matrix-green) !important;
    }

    /* GLOWING DATA TEXT */
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 15px var(--matrix-green);
        text-transform: uppercase;
    }
</style>

<script>
    // JS SOUNDSCAPE ENGINE
    const context = new (window.AudioContext || window.webkitAudioContext)();
    function playBeep(freq, type, duration) {
        const osc = context.createOscillator();
        const gain = context.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, context.currentTime);
        gain.gain.setValueAtTime(0.02, context.currentTime);
        osc.connect(gain);
        gain.connect(context.destination);
        osc.start();
        osc.stop(context.currentTime + duration);
    }

    document.addEventListener('click', () => playBeep(800, 'square', 0.05));
</script>
""", unsafe_allow_html=True)

# 3. STATE MANAGEMENT
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

# 4. VIEW: THE KASPERSKY SATELLITE GLOBE
def render_globe():
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>[ SATELLITE_LINK_ACQUISITION ]</h1>", unsafe_allow_html=True)
    
    # Arwes/Kaspersky-style 3D Arcs
    arc_data = pd.DataFrame([
        {'s': [-95, 37], 't': [53, 23], 'status': 'ACTIVE'}, # US to UAE
        {'s': [104, 35], 't': [78, 20], 'status': 'THREAT'}, # China to India
        {'s': [105, 61], 't': [45, 23], 'status': 'SUPPLY'}  # Russia to Saudi
    ])

    view = pdk.ViewState(latitude=30, longitude=60, zoom=1.2, pitch=45)
    
    layer = pdk.Layer(
        "ArcLayer", arc_data,
        get_source_position='s', get_target_position='t',
        get_source_color='[0, 255, 65, 120]', get_target_color='[0, 240, 255, 120]',
        width=3
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v11"), height=500)

    # TARGET GRID
    st.markdown("<div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; padding: 20px;'>", unsafe_allow_html=True)
    cols = st.columns(5)
    from config import COUNTRIES
    for i, (code, info) in enumerate(COUNTRIES.items()):
        if cols[i].button(f"NODE::{code}"):
            st.session_state.target = code
            st.rerun()

# 5. VIEW: THE MISSION IMPOSSIBLE DASHBOARD
def render_dashboard():
    target = st.session_state.target
    st.markdown(f"<h2 style='color:#00F0FF; padding: 20px;'>{target}_LINK_STABLE // 00{list(st.session_state.target).index(st.session_state.target[0])}</h2>", unsafe_allow_html=True)
    
    col_nav, col_center, col_side = st.columns([1, 3.5, 1.5])
    
    with col_nav:
        st.markdown("### SYSTEM")
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
        st.markdown("---")
        st.markdown("### MODULES")
        if st.button("🎯 TACTICAL_INTEL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("💬 DAEMON_V3"): st.session_state.active_tab = 'ai'; st.rerun()

    with col_center:
        if st.session_state.active_tab == 'intel':
            st.markdown("#### > VOLATILITY_WAVEFORM_ANALYSIS")
            # YOUR DATA LOAD HERE
            st.info("RECEIVING_DATA_STREAM...")
            # Simulate a glowing chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=[1,3,2,5,4], line=dict(color="#00F0FF", width=4, shape='spline'), fill='tozeroy'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("DAEMON_CONNECTED")
            st.chat_input("QUERY_TERMINAL...")

    with col_side:
        st.markdown("#### > THREAT_LOG")
        st.code("[08:00] NODE_LINK_ESTABLISHED\n[08:15] WTI_SHOCK_DETECTED\n[08:45] NPRS-1: BULLISH", language="bash")
        st.markdown("#### > RESOURCE_ALLOCATION")
        st.progress(85, "NPRS-1_ACCURACY")
        st.progress(12, "CRISIS_PROB")

# ROUTER
if st.session_state.target is None:
    render_globe()
else:
    render_dashboard()
