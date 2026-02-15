import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from datetime import datetime
import time
from modules.ai_engine import setup_rag_vector_db, get_ai_response
from config import COLORS, COUNTRIES

# 1. PAGE SETUP
st.set_page_config(page_title="OVIP // COMMAND", layout="wide", initial_sidebar_state="collapsed")

# 2. GLOBAL CSS (IRON MAN HUD STYLE)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg']};
        color: {COLORS['matrix']};
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important; 
    }}
    .block-container {{ padding: 1rem 2rem; max-width: 98% !important; }}
    
    /* CRT Scanline Overlay */
    .stApp::after {{
        content: " "; display: block; position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
                    linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 999; background-size: 100% 2px, 3px 100%; pointer-events: none;
    }}

    /* Futuristic Angled Panels */
    div[data-testid="column"] {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['matrix']};
        clip-path: polygon(0 0, calc(100% - 15px) 0, 100% 15px, 100% 100%, 15px 100%, 0 calc(100% - 15px));
        padding: 15px;
        box-shadow: inset 0 0 15px rgba(0, 255, 65, 0.1);
    }}

    /* Glowing Metrics */
    [data-testid="stMetricValue"] {{
        text-shadow: 0 0 10px {COLORS['matrix']};
        color: {COLORS['matrix']} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# 3. BOOT SEQUENCE
if 'booted' not in st.session_state:
    boot_screen = st.empty()
    with boot_screen.container():
        st.markdown("<h2 style='color:#00FF41; text-align:center;'>[ INITIALIZING OVIP_DAEMON ]</h2>", unsafe_allow_html=True)
        lines = ["> UPLINKING...", "> DECRYPTING...", "> ACCESS GRANTED."]
        log = ""
        for l in lines:
            log += l + "\n"; st.code(log, language="bash"); time.sleep(0.5)
    boot_screen.empty()
    st.session_state.booted = True

# 4. DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv('merged_final_corrected.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df_main = load_data()
vec, tfidf, rag_df = setup_rag_vector_db(df_main)

# 5. VIEW LOGIC
if 'target' not in st.session_state: st.session_state.target = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = 'intel'

# GLOBE VIEW
if st.session_state.target is None:
    st.markdown("<h2 style='text-align: center;'>[ TARGET_ACQUISITION_MODE ]</h2>", unsafe_allow_html=True)
    df_nodes = pd.DataFrame.from_dict(COUNTRIES, orient='index').reset_index()
    view = pdk.ViewState(latitude=20, longitude=50, zoom=1.1, pitch=45)
    layer = pdk.Layer("ScatterplotLayer", df_nodes, get_position="[lon, lat]", get_color="[0, 255, 65, 200]", get_radius=500000)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v10"))
    
    cols = st.columns(5)
    for i, (code, data) in enumerate(COUNTRIES.items()):
        if cols[i].button(f"{data['flag']} {code}"):
            st.session_state.target = code; st.rerun()

# DASHBOARD VIEW
else:
    target = st.session_state.target
    latest = df_main.iloc[-1]
    
    st.markdown(f"<h2 style='color:{COLORS['cyan']}; border-bottom:1px solid {COLORS['cyan']};'>NODE::{target} // {COUNTRIES[target]['name']}</h2>", unsafe_allow_html=True)
    
    c_nav, c_main = st.columns([1, 4])
    with c_nav:
        if st.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
        st.metric("WTI_PRICE", f"${latest['WTI']:.2f}")
        st.metric("VOL_SIGMA", f"{latest['Volatility']:.3f}")
        if st.button("🎯 INTEL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("💬 AI_DAEMON"): st.session_state.active_tab = 'ai'; st.rerun()

    with c_main:
        if st.session_state.active_tab == 'intel':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_main['Date'][-60:], y=df_main['Volatility'][-60:], line=dict(color=COLORS['cyan'], width=2), fill='tozeroy'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.active_tab == 'ai':
            chat_box = st.container(height=400)
            if prompt := st.chat_input("QUERY_SYSTEM..."):
                ans = get_ai_response(prompt, vec, tfidf, rag_df)
                chat_box.write(f"**USER:** {prompt}"); chat_box.write(f"**OVIP:** {ans}")
