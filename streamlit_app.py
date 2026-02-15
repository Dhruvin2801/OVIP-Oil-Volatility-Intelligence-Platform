import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from modules.ai_engine import setup_rag_vector_db, get_ai_response
from config import COLORS, COUNTRIES

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="OVIP // TACTICAL_OS", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. ARWES-STYLE CYBERPUNK CSS & SOUND ENGINE
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* BASE THEME */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000500 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    
    /* CRT SCANLINE SHADER */
    .stApp::after {
        content: " "; display: block; position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }

    /* NEON FRAMING FOR COLUMNS/CONTAINERS */
    [data-testid="stVerticalBlock"] > div {
        background: rgba(0, 20, 0, 0.4);
        border: 1px solid #00ff41;
        box-shadow: inset 0 0 15px rgba(0, 255, 65, 0.1);
        border-radius: 5px;
    }
    
    /* HACKER TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid #00ff41; }
    .stTabs [data-baseweb="tab"] { color: #00f0ff !important; font-family: 'Orbitron', sans-serif !important; }
    .stTabs [aria-selected="true"] { background: rgba(0, 255, 65, 0.15) !important; border: 1px solid #00ff41 !important; border-bottom: none !important;}

    /* GLOWING HEADERS & BUTTONS */
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif !important; text-shadow: 0 0 10px #00ff41; }
    .stButton>button {
        background: transparent !important; color: #00f0ff !important; border: 1px solid #00f0ff !important;
        font-family: 'Orbitron', sans-serif !important; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 15px #00f0ff; }
</style>

<script>
    // TACTICAL AUDIO BEEP ON CLICK
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {
        const o = actx.createOscillator(); const g = actx.createGain();
        o.type='square'; o.frequency.value=850;
        g.gain.value=0.03; o.connect(g); g.connect(actx.destination);
        o.start(); o.stop(actx.currentTime+0.05);
    });
</script>
""", unsafe_allow_html=True)

# ==========================================
# 3. STATE MANAGEMENT & DATA UPLINK
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

@st.cache_data
def load_data():
    for f in ['merged_fnal.csv', 'merged_final.csv', 'merged_final_corrected.csv']:
        if os.path.exists(f): 
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
    # Failsafe dummy data
    return pd.DataFrame({'Date':[pd.Timestamp.now()], 'WTI':[75.0], 'Volatility':[0.15], 'Crisis_Prob':[0.0]})

df_main = load_data()
vec, tfidf, rag_df = setup_rag_vector_db(df_main)

# ==========================================
# 4. VIEW 1: THE HOLOGRAPHIC 3D GLOBE
# ==========================================
if st.session_state.target is None:
    st.markdown("<h1 style='text-align:center;'>[ GLOBAL_THREAT_MATRIX ]</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#00f0ff;'>>> CLICK ON A TARGET NODE ON THE GLOBE TO INITIATE UPLINK</p>", unsafe_allow_html=True)
    
    # Extract Coordinates
    lats = [c['lat'] for c in COUNTRIES.values()]
    lons = [c['lon'] for c in COUNTRIES.values()]
    names = list(COUNTRIES.keys())
    
    # Build 3D Scattergeo Plot
    fig = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=14, color='#00f0ff', symbol='square', line=dict(width=2, color='#00ff41')),
        textfont = dict(family="Orbitron", size=18, color="#00ff41"),
        textposition = "top center", hoverinfo="text"
    ))
    
    # Style as Hologram Wireframe
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True, coastlinecolor="rgba(0, 255, 65, 0.4)",
        showland=True, landcolor="rgba(0, 15, 0, 0.8)",
        showocean=True, oceancolor="rgba(0, 0, 0, 1)",
        showcountries=True, countrycolor="rgba(0, 255, 65, 0.2)",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    # Render Interactive Globe (Click to Select)
    try:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
        if event and "selection" in event and event["selection"]["points"]:
            selected_idx = event["selection"]["points"][0]["point_index"]
            st.session_state.target = names[selected_idx]
            st.session_state.chat_history = [] # Reset chat for new country
            st.rerun()
    except:
        # Fallback if Streamlit version doesn't support on_select
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### MANUAL OVERRIDE:")
        cols = st.columns(5)
        for i, n in enumerate(names):
            if cols[i].button(n): st.session_state.target = n; st.session_state.chat_history = []; st.rerun()

# ==========================================
# 5. VIEW 2: THE COMMAND DASHBOARD
# ==========================================
else:
    target = st.session_state.target
    latest = df_main.iloc[-1]
    
    c_side, c_main = st.columns([1, 3])
    
    # -- SIDE PANEL: METRICS & CONTROLS --
    with c_side:
        st.markdown(f"### 📍 NODE::{target}")
        st.markdown("---")
        st.metric("WTI_INDEX", f"${latest.get('WTI', 0):.2f}")
        st.metric("VOL_SIGMA", f"{latest.get('Volatility', 0):.3f}")
        st.metric("CRISIS_PROB", f"{latest.get('Crisis_Prob', latest.get('Regime_Prob', 0)):.2f}")
        st.markdown("---")
        if st.button("🔴 DISCONNECT_LINK"):
            st.session_state.target = None
            st.rerun()

    # -- MAIN PANEL: TABS --
    with c_main:
        tab_intel, tab_ai = st.tabs(["[ 📊 TACTICAL_INTEL ]", "[ 🤖 DAEMON_AI_UPLINK ]"])
        
        # TAB 1: CHARTS
        with tab_intel:
            st.markdown("#### > VOLATILITY_WAVEFORM_ANALYSIS")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_main['Date'][-100:], y=df_main['Volatility'][-100:], 
                                      line=dict(color="#00f0ff", width=3), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'))
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        # TAB 2: AI CHAT INTERFACE
        with tab_ai:
            st.markdown(f"#### > ENCRYPTED COMMS OPEN: {target} REGION")
            
            # Render chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat Input
            if prompt := st.chat_input("TRANSMIT_QUERY..."):
                # Show user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.write(prompt)
                
                # Show AI response
                with st.chat_message("assistant"):
                    with st.spinner("DECRYPTING_RESPONSE..."):
                        # Inject country context into the AI query
                        country_prompt = f"Regarding {target}: {prompt}"
                        ans = get_ai_response(country_prompt, vec, tfidf, rag_df)
                        st.write(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
