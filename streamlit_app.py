import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import os
from modules.ai_engine import setup_rag_vector_db, get_ai_response
from config import COLORS, COUNTRIES

# 1. SYS INIT
st.set_page_config(page_title="OVIP // HUD", layout="wide", initial_sidebar_state="collapsed")

# 2. ARWES STYLE INJECTION
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Source+Code+Pro&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: #000 !important;
        color: {COLORS['matrix']} !important;
        font-family: 'Source Code Pro', monospace !important;
        overflow: hidden !important; /* NO SCROLLING */
        height: 100vh;
    }}

    /* ARWES VECTOR PANEL */
    .arwes-panel {{
        border: 1px solid {COLORS['matrix']};
        background: rgba(0, 40, 0, 0.2);
        padding: 20px;
        position: relative;
        clip-path: polygon(0 15px, 15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.1);
        height: 550px; /* LOCKS MIDDLE SPACE */
    }}

    /* GLOWING BUTTONS */
    .stButton > button {{
        background: transparent !important;
        border: 1px solid {COLORS['matrix']} !important;
        color: {COLORS['matrix']} !important;
        font-family: 'Orbitron' !important;
        text-transform: uppercase;
        border-radius: 0 !important;
        transition: 0.3s;
    }}
    .stButton > button:hover {{
        background: {COLORS['matrix']} !important;
        color: #000 !important;
        box-shadow: 0 0 20px {COLORS['matrix']};
    }}

    /* HIDE DEFAULTS */
    header, footer, #MainMenu {{ visibility: hidden; }}
</style>

<script>
    // TACTICAL SOUNDS
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {{
        const o = actx.createOscillator();
        const g = actx.createGain();
        o.type='square'; o.frequency.value=1000;
        g.gain.value=0.02; o.connect(g); g.connect(actx.destination);
        o.start(); o.stop(actx.currentTime+0.05);
    }});
</script>
""", unsafe_allow_html=True)

# 3. CORE LOGIC
if 'target' not in st.session_state: st.session_state.target = None
if 'tab' not in st.session_state: st.session_state.tab = 'intel'

# DATA
@st.cache_data
def load_sys_data():
    f = 'merged_fnal.csv'
    if os.path.exists(f): 
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame({'Date':[pd.Timestamp.now()], 'WTI':[0], 'Volatility':[0], 'Crisis_Prob':[0]})

df = load_sys_data()
vec, tfidf, rag_df = setup_rag_vector_db(df)

# 4. ROUTING
if st.session_state.target is None:
    # GLOBE VIEW
    st.markdown("<h1 style='text-align:center; font-family:Orbitron;'>[ TARGET_ACQUISITION ]</h1>", unsafe_allow_html=True)
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=20, longitude=50, zoom=1.5),
        map_style="mapbox://styles/mapbox/dark-v11",
        layers=[pdk.Layer("ArcLayer", data=[{'s':[-95,37], 't':[78,20]}], get_source_position='s', get_target_position='t', get_source_color=[0,255,65], get_target_color=[0,240,255], width=2)]
    ), height=500)
    
    cols = st.columns(5)
    for i, node in enumerate(COUNTRIES.keys()):
        if cols[i].button(f"LINK::{node}"): st.session_state.target = node; st.rerun()

else:
    # DASHBOARD VIEW
    st.markdown(f"<h2 style='color:#00f0ff; border-bottom:1px solid #00f0ff;'>NODE::{st.session_state.target}</h2>", unsafe_allow_html=True)
    
    c_nav, c_main = st.columns([1, 4])
    with c_nav:
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
        st.metric("VOL_SIGMA", f"{df.iloc[-1]['Volatility']:.3f}")
        if st.button("🎯 TACTICAL"): st.session_state.tab='intel'; st.rerun()
        if st.button("🤖 DAEMON"): st.session_state.tab='ai'; st.rerun()

    with c_main:
        # WRAP IN ARWES PANEL TO PREVENT BLANK MIDDLE
        st.markdown("<div class='arwes-panel'>", unsafe_allow_html=True)
        if st.session_state.tab == 'intel':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'][-60:], y=df['Volatility'][-60:], line=dict(color="#00f0ff", width=3), fill='tozeroy'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            if prompt := st.chat_input("QUERY_DAEMON..."):
                st.write(f"**OVIP:** {get_ai_response(prompt, vec, tfidf, rag_df)}")
        st.markdown("</div>", unsafe_allow_html=True)
