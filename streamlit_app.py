import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Safely import AI engine (failsafe if the module isn't found)
try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
# 'wide' layout is standard, but we'll use CSS to force the 1-page lock
st.set_page_config(page_title="OVIP // COMMAND_CENTER", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. THE "NO SCROLL" CSS & AUDIO ENGINE
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* THE NO-SCROLL LOCK */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000200 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important; /* PREVENTS SCROLLING */
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* REMOVE TOP BLANK SPACE */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    
    /* REMOVE DEFAULT STREAMLIT HEADER */
    header {visibility: hidden;}

    /* CRT Scanlines */
    .stApp::before {
        content: " "; display: block; position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }

    h1, h2, h3, h4, [data-testid="stMetricValue"] { 
        font-family: 'Orbitron', sans-serif !important; 
        text-shadow: 0 0 10px #00ff41; 
        text-transform: uppercase;
        margin-top: 0 !important;
    }
    
    [data-testid="stVerticalBlock"] > div.element-container { border-radius: 2px; }

    .stButton>button {
        background: rgba(0, 255, 65, 0.05) !important; color: #00f0ff !important; 
        border: 1px solid #00f0ff !important; font-family: 'Orbitron', sans-serif !important; 
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 15px #00f0ff; }
</style>

<script>
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {
        if (actx.state === 'suspended') actx.resume();
        const o = actx.createOscillator(); 
        const g = actx.createGain();
        o.type = 'square'; o.frequency.setValueAtTime(850, actx.currentTime); 
        g.gain.setValueAtTime(0.02, actx.currentTime); 
        o.connect(g); g.connect(actx.destination);
        o.start(); o.stop(actx.currentTime + 0.05); 
    });
</script>
""", unsafe_allow_html=True)

# ==========================================
# 3. DIRECTORY-AWARE DATA UPLINK
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
            except Exception: pass
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150), 'gpr': np.random.rand(150)*100})

df_main = load_data()

vec, tfidf, rag_df = None, None, None
if AI_AVAILABLE:
    try: vec, tfidf, rag_df = setup_rag_vector_db(df_main)
    except: pass

# ==========================================
# 4. ROUTER & DYNAMIC COUNTRY DATA
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None
if 'view_mode' not in st.session_state: st.session_state.view_mode = 'intel' # Toggle between Intel and Chat

COUNTRIES = {
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'event': 'Strategic Petroleum Reserve (SPR) refill initiated. Price floor hardening.', 'color': '#00f0ff'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'event': 'Industrial demand slowdown detected. Import quotas tightening.', 'color': '#FF003C'},
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'event': 'Refinery intake maxed. Russian crude discount shrinking.', 'color': '#00ff41'},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'MEDIUM', 'event': 'Fujairah bunkering volumes spike. Transit security heightened.', 'color': '#FFD700'},
    'SAUDI': {'lat': 23.89, 'lon': 45.08, 'risk': 'HIGH', 'event': 'OPEC+ voluntary cuts extended. Spare capacity acting as shock absorber.', 'color': '#FF003C'}
}

# ==========================================
# 5. VIEW 1: THE CLICKABLE GLOBE
# ==========================================
if st.session_state.target is None:
    st.markdown("<h2 style='text-align:center;'>[ GLOBAL_THREAT_MATRIX ]</h2>", unsafe_allow_html=True)
    
    lats, lons, names, colors = [], [], [], []
    for k, v in COUNTRIES.items():
        names.append(k); lats.append(v['lat']); lons.append(v['lon']); colors.append(v['color'])
    
    # The Clickable Globe
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=18, color=colors, symbol='square', line=dict(width=2, color='#fff')),
        textfont = dict(family="Orbitron", size=20, color="#fff"),
        textposition = "top center"
    ))
    
    fig_globe.update_geos(
        projection_type="orthographic", showcoastlines=True, coastlinecolor="rgba(0, 255, 65, 0.4)",
        showland=True, landcolor="rgba(0, 15, 0, 0.8)", showocean=True, oceancolor="rgba(0, 0, 0, 1)",
        showcountries=True, countrycolor="rgba(0, 255, 65, 0.2)", bgcolor="rgba(0,0,0,0)"
    )
    fig_globe.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    # Attempt native plotly click capture (works on newer Streamlit)
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
        st.rerun()

    # MANUAL OVERRIDE BAR (Failsafe)
    st.markdown("<p style='text-align:center; color:#00f0ff;'>IF 3D MAP UNRESPONSIVE, USE MANUAL UPLINK:</p>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, n in enumerate(names):
        if cols[i].button(f"[{n}]"): st.session_state.target = n; st.rerun()

# ==========================================
# 6. VIEW 2: DYNAMIC COUNTRY DASHBOARD
# ==========================================
else:
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    
    # --- TOP ROW: DYNAMIC HEADER ---
    c_head1, c_head2 = st.columns([4, 1])
    c_head1.markdown(f"<h2 style='color:{intel['color']};'>NODE::{target} UPLINK ACTIVE</h2>", unsafe_allow_html=True)
    if c_head2.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()
    
    st.markdown(f"<div style='border-top: 2px solid {intel['color']}; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # --- ROW 2: DYNAMIC METRICS ---
    # We modify metrics slightly based on country to make it feel "dynamic"
    mod = 1.05 if intel['risk'] == 'HIGH' else (0.95 if intel['risk'] == 'LOW' else 1.0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("REGIONAL_WTI_PREMIUM", f"${(latest.get('WTI', 75) * mod):.2f}")
    c2.metric("LOCAL_VOL_SIGMA", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    c3.metric("GEOPOLITICAL_RISK (GPR)", f"{latest.get('gpr', 50):.1f}")
    c4.metric(f"{target}_THREAT_LEVEL", intel['risk'])

    # --- ROW 3: SPLIT VIEW (CHART + INTEL/CHAT) ---
    col_chart, col_side = st.columns([2.5, 1])

    with col_chart:
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Dynamic Volatility Line
        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Adjusted Vol',
            line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=f'rgba({255 if intel["risk"]=="HIGH" else 0}, 255, 65, 0.1)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)',
            line=dict(color="#00ff41", width=2, dash='dot')
        ), secondary_y=True)

        # Plotly height locked to fit screen perfectly without scroll
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)')
        fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI Price", color="#00ff41", showgrid=False, secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # Toggle Button for Pop-up Chat functionality
        if st.session_state.view_mode == 'intel':
            if st.button("💬 OPEN_AI_TERMINAL"): st.session_state.view_mode = 'chat'; st.rerun()
            
            # Dynamic Country Log
            st.markdown(f"<div style='background:rgba(0,20,0,0.6); padding:15px; border:1px solid {intel['color']}; height: 320px;'>", unsafe_allow_html=True)
            st.markdown(f"### > {target} INTEL LOG")
            st.markdown(f"""
            <ul style='color: #00ff41; font-family: Share Tech Mono; font-size: 0.9em;'>
                <li><b style='color:{intel['color']};'>[LIVE_EVENT]:</b> {intel['event']}</li>
                <br>
                <li><b>[GPR_MONITOR]:</b> Localized index tracking at {latest.get('gpr', 50):.1f}.</li>
                <br>
                <li><b>[SYSTEM_PROJECTION]:</b> Volatility expected to {'escalate' if intel['risk']=='HIGH' else 'stabilize'} over next 72 hours.</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            if st.button("📊 BACK_TO_INTEL"): st.session_state.view_mode = 'intel'; st.rerun()
            
            # The Chat Pop-Up (Replaces the Intel Log)
            st.markdown(f"<div style='background:rgba(0,20,0,0.6); padding:10px; border:1px solid #00f0ff; height: 320px;'>", unsafe_allow_html=True)
            if "chat_log" not in st.session_state:
                st.session_state.chat_log = [{"role": "assistant", "content": f"DAEMON ONLINE. State your query regarding {target}."}]

            # Scrollable chat box
            chat_box = st.container(height=210, border=False)
            with chat_box:
                for msg in st.session_state.chat_log:
                    c = "#00f0ff" if msg['role'] == 'user' else "#00ff41"
                    n = "USER" if msg['role'] == 'user' else "DAEMON"
                    st.markdown(f"<span style='color:{c}; font-size:0.85em;'><b>{n}:~$</b> {msg['content']}</span><br>", unsafe_allow_html=True)

            if prompt := st.chat_input("TRANSMIT..."):
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                if AI_AVAILABLE and vec is not None:
                    # Inject country context into prompt
                    context_prompt = f"In the context of the region {target}: {prompt}"
                    ans = get_ai_response(context_prompt, vec, tfidf, rag_df)
                else:
                    ans = "SYSTEM OFFLINE."
                st.session_state.chat_log.append({"role": "assistant", "content": ans})
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
