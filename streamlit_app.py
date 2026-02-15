import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# 1. INITIALIZE SYSTEM
st.set_page_config(page_title="OVIP // ARWES_ACTV", layout="wide", initial_sidebar_state="collapsed")

# 2. ARWES SHADER & SOUND ENGINE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Source+Code+Pro:wght@300;400&display=swap');

    :root {
        --arwes-green: #00ff41;
        --arwes-cyan: #00f0ff;
        --arwes-bg: #000800;
        --glow: 0 0 15px rgba(0, 255, 65, 0.4);
    }

    /* FULLSCREEN LOCK */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--arwes-bg) !important;
        color: var(--arwes-green) !important;
        font-family: 'Source Code Pro', monospace !important;
        overflow: hidden !important;
        height: 100vh;
    }

    /* ARWES VECTOR FRAME SYSTEM */
    .arwes-frame {
        border: 1px solid var(--arwes-green);
        background: rgba(0, 20, 0, 0.6);
        position: relative;
        padding: 20px;
        clip-path: polygon(
            0 20px, 20px 0, 
            calc(100% - 20px) 0, 100% 20px, 
            100% calc(100% - 20px), calc(100% - 20px) 100%, 
            20px 100%, 0 calc(100% - 20px)
        );
        box-shadow: var(--glow), inset 0 0 20px rgba(0, 255, 65, 0.1);
        height: 100%;
    }

    /* GRID OVERLAY */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: radial-gradient(rgba(0, 255, 65, 0.1) 1px, transparent 0);
        background-size: 30px 30px;
        z-index: -1;
    }

    /* SCANLINES */
    .stApp::after {
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%);
        background-size: 100% 4px;
        pointer-events: none; z-index: 1000;
    }

    /* GLOWING BUTTONS */
    .stButton > button {
        background: transparent !important;
        color: var(--arwes-green) !important;
        border: 1px solid var(--arwes-green) !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: rgba(0, 255, 65, 0.2) !important;
        box-shadow: 0 0 20px var(--arwes-green) !important;
    }
</style>

<script>
    // ARWES TACTICAL AUDIO ENGINE
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function playClick(freq=800, type='square', vol=0.03) {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(vol, audioCtx.currentTime);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.05);
    }
    document.addEventListener('click', () => playClick());
</script>
""", unsafe_allow_html=True)

# 3. STATE & DATA
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

# 4. VIEW: THE GLOBE (Target Acquisition)
def render_globe():
    st.markdown("<h1 style='text-align:center; font-family:Orbitron; letter-spacing:10px;'>[ TARGETING_SYSTEM ]</h1>", unsafe_allow_html=True)
    
    # 3D Kaspersky-Style Arcs
    view = pdk.ViewState(latitude=20, longitude=50, zoom=1.5, pitch=45)
    st.pydeck_chart(pdk.Deck(
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v11",
        layers=[pdk.Layer("ArcLayer", data=[{'s':[-95,37], 't':[53,23]}], get_source_position='s', get_target_position='t', get_source_color=[0,255,65,150], get_target_color=[0,240,255,150], width=4)]
    ), height=550)

    cols = st.columns(5)
    nodes = ["USA", "CHINA", "INDIA", "UAE", "SAUDI"]
    for i, node in enumerate(nodes):
        if cols[i].button(f"LINK::{node}"):
            st.session_state.target = node
            st.rerun()

# 5. VIEW: THE TACTICAL DASHBOARD
def render_dashboard():
    target = st.session_state.target
    
    # HUD HEADER
    st.markdown(f"""
        <div style='display:flex; justify-content:space-between; border-bottom:1px solid #00ff41; padding:10px;'>
            <h2 style='font-family:Orbitron; margin:0;'>NODE_LINK // {target}</h2>
            <div style='color:#00f0ff;'>STATUS: ENCRYPTED_UPLINK_STABLE</div>
        </div>
    """, unsafe_allow_html=True)

    # GRID LAYOUT (The fix for "blank middle")
    c_side, c_main = st.columns([1, 4])

    with c_side:
        st.markdown("### MODULES")
        if st.button("INTEL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("DAEMON"): st.session_state.active_tab = 'ai'; st.rerun()
        st.markdown("---")
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()

    with c_main:
        # We wrap the main content in a container with a fixed height
        with st.container(height=500, border=False):
            if st.session_state.active_tab == 'intel':
                st.markdown("#### > TACTICAL_WAVEFORM_ANALYSIS")
                # Placeholder for your Volatility graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=[10,15,13,17], line=dict(color="#00f0ff", width=4), fill='tozeroy'))
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("DAEMON_V3_ONLINE")
                st.chat_input("COMMAND_STRING...")

# ROUTER
if st.session_state.target is None:
    render_globe()
else:
    render_dashboard()
