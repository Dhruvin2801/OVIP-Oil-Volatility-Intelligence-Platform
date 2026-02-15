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

# 2. ARWES UI ENGINE (CSS Grid + Audio + Scanlines)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* 100% VIEWPORT LOCK - NO SCROLLING */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg']} !important;
        color: {COLORS['matrix']} !important;
        font-family: 'Share Tech Mono', monospace !important;
        height: 100vh !important;
        overflow: hidden !important;
    }}
    .main .block-container {{
        padding: 1rem 1.5rem !important;
        max-width: 99% !important;
        height: 100vh !important;
    }}

    /* ARWES VECTOR PANELS (Iron Man Style) */
    div[data-testid="column"] {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['matrix']};
        clip-path: polygon(0 15px, 15px 0, 100% 0, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0 100%);
        padding: 20px;
        box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.1), 0 0 10px rgba(0, 255, 65, 0.2);
    }}

    /* SCANLINE SHADER */
    .stApp::after {{
        content: " "; display: block; position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.15) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }}

    /* GLOWING HEADERS */
    h1, h2, h3, [data-testid="stMetricValue"] {{
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px {COLORS['matrix']};
        text-transform: uppercase;
    }}

    /* HIDE STREAMLIT BRANDING */
    #MainMenu, footer, header {{ visibility: hidden; }}
</style>

<script>
    // TACTICAL SOUND ENGINE
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function playChirp() {{
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.type = 'square'; osc.frequency.setValueAtTime(850, ctx.currentTime);
        g.gain.setValueAtTime(0.02, ctx.currentTime);
        osc.connect(g); g.connect(ctx.destination);
        osc.start(); osc.stop(ctx.currentTime + 0.05);
    }}
    document.addEventListener('click', playChirp);
</script>
""", unsafe_allow_html=True)

# 3. STATE & DATA LOADING
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

@st.cache_data
def load_data():
    # Priority search for the CSV
    for f in ['merged_fnal.csv', 'merged_final.csv', 'merged_final_corrected.csv']:
        if os.path.exists(f):
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    # Fallback data
    return pd.DataFrame({'Date':[pd.Timestamp.now()],'WTI':[75.0],'Volatility':[0.15],'Crisis_Prob':[0.0]})

df_main = load_data()
vec, tfidf, rag_df = setup_rag_vector_db(df_main)

# 4. VIEW: SATELLITE TARGET ACQUISITION
if st.session_state.target is None:
    st.markdown("<h1 style='text-align:center; letter-spacing:8px;'>[ OVIP // GLOBAL_THREAT_UPLINK ]</h1>", unsafe_allow_html=True)
    
    # 3D Arc Globe
    view = pdk.ViewState(latitude=20, longitude=55, zoom=1.4, pitch=40)
    st.pydeck_chart(pdk.Deck(
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/dark-v11",
        layers=[pdk.Layer("ArcLayer", data=[{'s':[-95,37], 't':[53,23]}], get_source_position='s', get_target_position='t', get_source_color=[0,255,65,150], get_target_color=[0,240,255,150], width=3)]
    ), height=550)
    
    st.markdown("<div style='display:flex; justify-content:center; gap:10px;'>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, node in enumerate(COUNTRIES.keys()):
        if cols[i].button(f"NODE::{node}"):
            st.session_state.target = node; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# 5. VIEW: TACTICAL HUD
else:
    target = st.session_state.target
    latest = df_main.iloc[-1]
    
    # Header HUD
    st.markdown(f"""
        <div style='display:flex; justify-content:space-between; border-bottom:1px solid #00f0ff; padding:10px; margin-bottom:15px;'>
            <h2 style='color:#00f0ff; margin:0;'>LINK_ESTABLISHED // NODE_{target}</h2>
            <div style='color:#00ff41;'>UPLINK_SECURE | {datetime.now().strftime('%H:%M:%S')} UTC</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_nav, col_main = st.columns([1, 4])
    
    with col_nav:
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
        st.markdown("---")
        st.metric("VOL_SIGMA", f"{latest.get('Volatility', 0):.3f}")
        st.metric("WTI_PRICE", f"${latest.get('WTI', 0):.2f}")
        st.markdown("---")
        if st.button("🎯 TACTICAL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("🤖 DAEMON"): st.session_state.active_tab = 'ai'; st.rerun()

    with col_main:
        # Main Hud Frame
        with st.container(height=520, border=False):
            if st.session_state.active_tab == 'intel':
                st.markdown("#### > TACTICAL_VOLATILITY_WAVEFORM")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_main['Date'][-60:], y=df_main['Volatility'][-60:], line=dict(color="#00f0ff", width=4), fill='tozeroy'))
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("#### > DAEMON_V3_SECURE_TERMINAL")
                if prompt := st.chat_input("QUERY_DAEMON..."):
                    ans = get_ai_response(prompt, vec, tfidf, rag_df)
                    st.write(f"**USER:** {prompt}")
                    st.write(f"**DAEMON:** {ans}")
