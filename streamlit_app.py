import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import time
import os
from modules.ai_engine import setup_rag_vector_db, get_ai_response
from config import COLORS, COUNTRIES

# 1. INITIALIZE TACTICAL OS
st.set_page_config(page_title="OVIP // HUD_v3", layout="wide", initial_sidebar_state="collapsed")

# 2. ARWES UI ENGINE (CSS Grid + Audio)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* 100% Viewport Lock - NO SCROLLING */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg']} !important;
        color: {COLORS['matrix']} !important;
        font-family: 'Share Tech Mono', monospace !important;
        height: 100vh !important;
        overflow: hidden !important;
    }}
    .main .block-container {{
        padding: 1rem 1rem !important;
        max-width: 99% !important;
        height: 100vh !important;
    }}

    /* ARWES VECTOR PANELS */
    .st-emotion-cache-1wivap2, div[data-testid="stVerticalBlock"] > div.element-container {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['matrix']};
        clip-path: polygon(0 15px, 15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
        padding: 10px;
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.1);
        margin-bottom: 0px !important;
    }}

    /* GLOWING HEADERS */
    h1, h2, h3, [data-testid="stMetricValue"] {{
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px {COLORS['matrix']};
    }}

    /* HIDE STREAMLIT BRANDING */
    #MainMenu, footer, header {{ visibility: hidden; }}
</style>

<script>
    // TACTICAL CLICK FEEDBACK
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {{
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.type = 'square'; osc.frequency.setValueAtTime(900, ctx.currentTime);
        g.gain.setValueAtTime(0.02, ctx.currentTime);
        osc.connect(g); g.connect(ctx.destination);
        osc.start(); osc.stop(ctx.currentTime + 0.05);
    }});
</script>
""", unsafe_allow_html=True)

# 3. STATE & DATA
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

@st.cache_data
def load_data():
    for f in ['merged_fnal.csv', 'merged_final.csv']:
        if os.path.exists(f): return pd.read_csv(f)
    return pd.DataFrame({'Date':['2025-01-01'],'WTI':[75],'Volatility':[0.12]})

df_main = load_data()
df_main['Date'] = pd.to_datetime(df_main['Date'])
vec, tfidf, rag_df = setup_rag_vector_db(df_main)

# 4. VIEW LOGIC
if st.session_state.target is None:
    # --- GLOBE VIEW ---
    st.markdown("<h1 style='text-align:center;'>[ OVIP // TARGET_ACQUISITION ]</h1>", unsafe_allow_html=True)
    view = pdk.ViewState(latitude=20, longitude=50, zoom=1.5, pitch=45)
    st.pydeck_chart(pdk.Deck(initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v11",
                             layers=[pdk.Layer("ArcLayer", data=[{'s':[-95,37], 't':[104,35]}], get_source_position='s', get_target_position='t', get_source_color=[0,255,65], get_target_color=[0,240,255], width=2)]), height=600)
    
    cols = st.columns(5)
    for i, node in enumerate(COUNTRIES.keys()):
        if cols[i].button(f"LINK::{node}"): st.session_state.target = node; st.rerun()

else:
    # --- DASHBOARD VIEW ---
    latest = df_main.iloc[-1]
    st.markdown(f"<h2 style='color:#00f0ff; border-bottom:1px solid #00f0ff;'>NODE::{st.session_state.target}</h2>", unsafe_allow_html=True)
    
    # LOCK CONTENT TO 80% SCREEN HEIGHT
    col_nav, col_main = st.columns([1, 4])
    
    with col_nav:
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
        st.metric("VOL_SIGMA", f"{latest['Volatility']:.3f}")
        st.metric("CRISIS_PROB", "0.00")
        if st.button("🎯 INTEL"): st.session_state.active_tab='intel'; st.rerun()
        if st.button("🤖 DAEMON"): st.session_state.active_tab='ai'; st.rerun()

    with col_main:
        with st.container(height=550, border=False):
            if st.session_state.active_tab == 'intel':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_main['Date'][-60:], y=df_main['Volatility'][-60:], line=dict(color="#00f0ff", width=3), fill='tozeroy'))
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.chat_input("COMMAND_STRING...")
