import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="OVIP // COMMAND_CENTER", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. AGGRESSIVE NO-SCROLL CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* KILL ALL PADDING AND LOCK SCROLLING */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000200 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
        overflow: hidden !important; 
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* ELIMINATE THE TOP WHITE SPACE */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        margin-top: -50px !important; /* Forces the UI up */
        max-width: 100% !important;
    }
    
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
    
    .stButton>button {
        background: rgba(0, 255, 65, 0.05) !important; color: #00f0ff !important; 
        border: 1px solid #00f0ff !important; font-family: 'Orbitron', sans-serif !important; 
        width: 100%; padding: 2px !important; font-size: 0.8em !important; transition: 0.2s;
    }
    .stButton>button:hover { background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 15px #00f0ff; }
</style>

<script>
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {
        if (actx.state === 'suspended') actx.resume();
        const o = actx.createOscillator(); const g = actx.createGain();
        o.type = 'square'; o.frequency.setValueAtTime(900, actx.currentTime); 
        g.gain.setValueAtTime(0.015, actx.currentTime); 
        o.connect(g); g.connect(actx.destination);
        o.start(); o.stop(actx.currentTime + 0.04); 
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
# 4. 30 COUNTRIES & DYNAMIC INTEL
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None
if 'view_mode' not in st.session_state: st.session_state.view_mode = 'intel'

# 30 Countries spanning major geopolitics
COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'event': 'Refinery intake stable. Russian crude discount shrinking.', 'color': '#00ff41', 'mod': 0.95},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'event': 'SPR refill operations ongoing. Shale production flat.', 'color': '#00f0ff', 'mod': 1.0},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'event': 'Industrial demand slowdown. Import quotas tightening.', 'color': '#FF003C', 'mod': 1.1},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'event': 'Sanction evasion routing detected via dark fleet.', 'color': '#FF003C', 'mod': 1.25},
    'SAUDI': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'event': 'OPEC+ voluntary cuts maintained. Spare capacity high.', 'color': '#FFD700', 'mod': 1.0},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'event': 'Fujairah bunkering volumes spike. Storage expanding.', 'color': '#00ff41', 'mod': 0.98},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'event': 'Strait of Hormuz transit harassment reported.', 'color': '#FF003C', 'mod': 1.3},
    'VENEZUELA': {'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'event': 'Production stagnating due to lack of capital expenditure.', 'color': '#FF003C', 'mod': 1.15},
    'BRAZIL': {'lat': -14.23, 'lon': -51.92, 'risk': 'LOW', 'event': 'Pre-salt offshore production hitting record highs.', 'color': '#00ff41', 'mod': 0.9},
    'UK': {'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'event': 'North Sea windfall tax impacts future drilling plans.', 'color': '#00f0ff', 'mod': 1.02},
    'NORWAY': {'lat': 60.47, 'lon': 8.46, 'risk': 'LOW', 'event': 'Equinor gas exports to Europe maxed. Secure supply.', 'color': '#00ff41', 'mod': 0.9},
    'NIGERIA': {'lat': 9.08, 'lon': 8.67, 'risk': 'HIGH', 'event': 'Pipeline vandalism causes force majeure on Bonny Light.', 'color': '#FF003C', 'mod': 1.2},
    'ANGOLA': {'lat': -11.20, 'lon': 17.87, 'risk': 'MEDIUM', 'event': 'Deepwater asset decline rates accelerating.', 'color': '#FFD700', 'mod': 1.05},
    'LIBYA': {'lat': 26.33, 'lon': 17.22, 'risk': 'CRITICAL', 'event': 'Eastern faction blockades export terminals.', 'color': '#FF003C', 'mod': 1.4},
    'IRAQ': {'lat': 33.22, 'lon': 43.67, 'risk': 'HIGH', 'event': 'Federal vs Kurdish export disputes unresolved.', 'color': '#FF003C', 'mod': 1.15},
    'KUWAIT': {'lat': 29.31, 'lon': 47.48, 'risk': 'LOW', 'event': 'Al-Zour refinery running at full capacity.', 'color': '#00ff41', 'mod': 0.95},
    'QATAR': {'lat': 25.35, 'lon': 51.18, 'risk': 'LOW', 'event': 'North Field LNG expansion on schedule.', 'color': '#00ff41', 'mod': 0.95},
    'CANADA': {'lat': 56.13, 'lon': -106.34, 'risk': 'LOW', 'event': 'Trans Mountain pipeline expansion easing bottlenecks.', 'color': '#00ff41', 'mod': 0.98},
    'MEXICO': {'lat': 23.63, 'lon': -102.55, 'risk': 'MEDIUM', 'event': 'Pemex debt restructuring causing operational delays.', 'color': '#FFD700', 'mod': 1.05},
    'GERMANY': {'lat': 51.16, 'lon': 10.45, 'risk': 'MEDIUM', 'event': 'Macro slowdown reducing middle-distillate demand.', 'color': '#00f0ff', 'mod': 1.0},
    'JAPAN': {'lat': 36.20, 'lon': 138.25, 'risk': 'MEDIUM', 'event': 'Nuclear restart program slowing LNG imports slightly.', 'color': '#00f0ff', 'mod': 1.0},
    'SOUTH KOREA': {'lat': 35.90, 'lon': 127.76, 'risk': 'LOW', 'event': 'Strategic reserves fully stocked.', 'color': '#00ff41', 'mod': 0.95},
    'AUSTRALIA': {'lat': -25.27, 'lon': 133.77, 'risk': 'LOW', 'event': 'Gorgon LNG facility maintenance completed.', 'color': '#00ff41', 'mod': 0.95},
    'ALGERIA': {'lat': 28.03, 'lon': 1.65, 'risk': 'MEDIUM', 'event': 'European gas supply contracts renegotiated.', 'color': '#FFD700', 'mod': 1.0},
    'EGYPT': {'lat': 26.82, 'lon': 30.80, 'risk': 'HIGH', 'event': 'Suez Canal transit risks remain elevated due to regional conflict.', 'color': '#FF003C', 'mod': 1.15},
    'TURKEY': {'lat': 38.96, 'lon': 35.24, 'risk': 'MEDIUM', 'event': 'Bosphorus transit fees adjusted. Russian imports high.', 'color': '#FFD700', 'mod': 1.05},
    'SOUTH AFRICA': {'lat': -30.55, 'lon': 22.93, 'risk': 'MEDIUM', 'event': 'Cape of Good Hope rerouting adding 14 days to freight.', 'color': '#FFD700', 'mod': 1.1},
    'SINGAPORE': {'lat': 1.35, 'lon': 103.81, 'risk': 'LOW', 'event': 'Malacca Strait maritime traffic flowing normally.', 'color': '#00ff41', 'mod': 0.95},
    'INDONESIA': {'lat': -0.78, 'lon': 113.92, 'risk': 'MEDIUM', 'event': 'Domestic subsidy policies straining fiscal budget.', 'color': '#FFD700', 'mod': 1.0},
    'OMAN': {'lat': 21.51, 'lon': 55.92, 'risk': 'LOW', 'event': 'Gulf of Oman crude blending operations expanding.', 'color': '#00ff41', 'mod': 0.95}
}

# ==========================================
# 5. VIEW 1: THE CLICKABLE GLOBE
# ==========================================
if st.session_state.target is None:
    st.markdown("<h2 style='text-align:center; padding-top:20px;'>[ GLOBAL_THREAT_MATRIX ]</h2>", unsafe_allow_html=True)
    
    lats, lons, names, colors = [], [], [], []
    for k, v in COUNTRIES.items():
        names.append(k); lats.append(v['lat']); lons.append(v['lon']); colors.append(v['color'])
    
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=12, color=colors, symbol='square', line=dict(width=1, color='#000')),
        textfont = dict(family="Share Tech Mono", size=12, color=colors),
        textposition = "top center"
    ))
    
    # FACE INDIA BY DEFAULT (lon: 78, lat: 20)
    fig_globe.update_geos(
        projection_type="orthographic", showcoastlines=True, coastlinecolor="rgba(0, 255, 65, 0.2)",
        showland=True, landcolor="rgba(0, 15, 0, 0.8)", showocean=True, oceancolor="rgba(0, 0, 0, 1)",
        showcountries=True, countrycolor="rgba(0, 255, 65, 0.1)", bgcolor="rgba(0,0,0,0)",
        center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
    )
    fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
        st.rerun()

    # TACTICAL FALLBACK GRID (30 Nodes)
    st.markdown("<p style='text-align:center; color:#00f0ff; font-size: 0.8em;'>MANUAL UPLINK OVERRIDE:</p>", unsafe_allow_html=True)
    cols = st.columns(10)
    for i, n in enumerate(names):
        if cols[i%10].button(n): st.session_state.target = n; st.rerun()

# ==========================================
# 6. VIEW 2: DYNAMIC COUNTRY DASHBOARD
# ==========================================
else:
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    st.markdown(f"<h2 style='color:{intel['color']}; padding-top:10px;'>NODE::{target} UPLINK ACTIVE</h2>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,0.5])
    c1.metric("REGIONAL_WTI_ADJUSTED", f"${(latest.get('WTI', 75) * mod):.2f}")
    c2.metric("LOCAL_VOL_SIGMA", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    c3.metric("GEOPOLITICAL_RISK (GPR)", f"{(latest.get('gpr', 50) * mod):.1f}")
    c4.metric("THREAT_LEVEL", intel['risk'])
    if c5.button("🔴 DISCONNECT"): st.session_state.target = None; st.rerun()

    st.markdown(f"<div style='border-top: 1px solid {intel['color']}; margin-bottom: 5px;'></div>", unsafe_allow_html=True)

    # --- ROW 3: SPLIT VIEW (CHART + INTEL/CHAT) ---
    col_chart, col_side = st.columns([2.5, 1])

    with col_chart:
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Graph color changes based on the country you clicked
        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Volatility',
            line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=f'rgba({255 if intel["risk"] in ["HIGH", "CRITICAL"] else 0}, {255 if intel["risk"]=="LOW" else 0}, 65, 0.1)'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)',
            line=dict(color="#ffffff", width=1, dash='dot')
        ), secondary_y=True)

        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)')
        fig.update_yaxes(title_text=f"{target} Local Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="Global WTI Price", color="#ffffff", showgrid=False, secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        # THE TOGGLE (INTEL <--> CHAT)
        if st.session_state.view_mode == 'intel':
            if st.button("💬 OPEN_AI_TERMINAL"): st.session_state.view_mode = 'chat'; st.rerun()
            
            st.markdown(f"<div style='background:rgba(0,20,0,0.6); padding:10px; border:1px solid {intel['color']}; height: 350px;'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:{intel['color']};'> > {target} INTEL LOG</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <ul style='color: #00ff41; font-family: Share Tech Mono; font-size: 0.9em;'>
                <li><b style='color:{intel['color']};'>[LIVE_EVENT]:</b> {intel['event']}</li><br>
                <li><b>[GPR_MONITOR]:</b> Localized index tracking at {(latest.get('gpr', 50)*mod):.1f}.</li><br>
                <li><b>[SYSTEM_PROJECTION]:</b> Volatility expected to {'escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else 'stabilize'} over next 72 hours.</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            if st.button("📊 BACK_TO_INTEL"): st.session_state.view_mode = 'intel'; st.rerun()
            
            st.markdown(f"<div style='background:rgba(0,20,0,0.6); padding:5px; border:1px solid #00f0ff; height: 350px;'>", unsafe_allow_html=True)
            if "chat_log" not in st.session_state:
                st.session_state.chat_log = [{"role": "assistant", "content": f"DAEMON ONLINE. Query regarding {target}."}]

            # Mini Chat window embedded perfectly
            chat_box = st.container(height=240, border=False)
            with chat_box:
                for msg in st.session_state.chat_log:
                    c = "#00f0ff" if msg['role'] == 'user' else "#00ff41"
                    n = "USER" if msg['role'] == 'user' else "DAEMON"
                    st.markdown(f"<span style='color:{c}; font-size:0.85em;'><b>{n}:~$</b> {msg['content']}</span><br>", unsafe_allow_html=True)

            if prompt := st.chat_input("TRANSMIT..."):
                st.session_state.chat_log.append({"role": "user", "content": prompt})
                if AI_AVAILABLE and vec is not None:
                    context_prompt = f"In the context of the region {target}: {prompt}"
                    ans = get_ai_response(context_prompt, vec, tfidf, rag_df)
                else: ans = "SYSTEM OFFLINE."
                st.session_state.chat_log.append({"role": "assistant", "content": ans})
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
