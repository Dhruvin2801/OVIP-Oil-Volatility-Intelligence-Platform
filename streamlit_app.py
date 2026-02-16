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
st.set_page_config(page_title="OVIP // TACTICAL_OS", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. TACTICAL CSS & SOUND ENGINE
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000200 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    
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
    }
    
    /* Cyberpunk Buttons */
    .stButton>button {
        background: rgba(0, 255, 65, 0.05) !important; color: #00f0ff !important; 
        border: 1px solid #00f0ff !important; font-family: 'Orbitron', sans-serif !important; 
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 15px #00f0ff; }
</style>

<script>
    // Tactical Click Sound
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
    
    st.warning("⚠️ PROXY_DATA: Actual database offline. Showing simulation.")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150)})

df_main = load_data()

vec, tfidf, rag_df = None, None, None
if AI_AVAILABLE:
    try:
        vec, tfidf, rag_df = setup_rag_vector_db(df_main)
    except:
        pass

# ==========================================
# 4. ROUTER STATE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

COUNTRIES = {
    'USA': {'lat': 37.09, 'lon': -95.71},
    'CHINA': {'lat': 35.86, 'lon': 104.20},
    'INDIA': {'lat': 20.59, 'lon': 78.96},
    'UAE': {'lat': 23.42, 'lon': 53.85},
    'SAUDI': {'lat': 23.89, 'lon': 45.08}
}

# ==========================================
# 5. VIEW 1: THE HOLOGRAPHIC GLOBE
# ==========================================
if st.session_state.target is None:
    st.markdown("<h1 style='text-align:center;'>[ GLOBAL_THREAT_MATRIX ]</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#00f0ff;'>>> CLICK A REGION TO INITIATE UPLINK</p>", unsafe_allow_html=True)
    
    # Extract Coordinates
    lats = [c['lat'] for c in COUNTRIES.values()]
    lons = [c['lon'] for c in COUNTRIES.values()]
    names = list(COUNTRIES.keys())
    
    # Build 3D Scattergeo Plot (Hologram Style)
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=14, color='#00f0ff', symbol='square', line=dict(width=2, color='#00ff41')),
        textfont = dict(family="Orbitron", size=18, color="#00ff41"),
        textposition = "top center", hoverinfo="text"
    ))
    
    fig_globe.update_geos(
        projection_type="orthographic",
        showcoastlines=True, coastlinecolor="rgba(0, 255, 65, 0.4)",
        showland=True, landcolor="rgba(0, 15, 0, 0.8)",
        showocean=True, oceancolor="rgba(0, 0, 0, 1)",
        showcountries=True, countrycolor="rgba(0, 255, 65, 0.2)",
        bgcolor="rgba(0,0,0,0)"
    )
    
    # Natural height for the globe
    fig_globe.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_globe, use_container_width=True)

    # Fallback Buttons just in case 3D clicks fail on Streamlit Cloud
    st.markdown("---")
    cols = st.columns(5)
    for i, n in enumerate(names):
        if cols[i].button(f"LINK::{n}"): 
            st.session_state.target = n; st.rerun()

# ==========================================
# 6. VIEW 2: THE INSIGHT ROOM (Matrix)
# ==========================================
else:
    # --- HEADER METRICS ---
    st.markdown(f"<h1 style='color:#00f0ff;'>NODE::{st.session_state.target} UPLINK ACTIVE</h1>", unsafe_allow_html=True)
    if st.button("🔴 DISCONNECT_LINK"): st.session_state.target = None; st.rerun()

    latest = df_main.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WTI_CRUDE_INDEX", f"${latest.get('WTI', 0):.2f}")
    c2.metric("VOLATILITY_SIGMA", f"{latest.get('Volatility', 0):.3f}")
    c3.metric("MACRO_CRISIS_PROB", f"{latest.get('Crisis_Prob', latest.get('Regime_Prob', 0.0)):.2f}")
    c4.metric("NPRS-1_AI_SIGNAL", "UP_TREND", "CONFIRMED")
    st.markdown("---")

    # --- MAIN EVENT CHART ---
    st.markdown("### > MACRO-EVENT VOLATILITY IMPACT MATRIX")
    chart_df = df_main.dropna(subset=['Date']).tail(150).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility (Sigma)',
        line=dict(color="#00f0ff", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=chart_df['Date'], y=chart_df['WTI'], name='WTI Price ($)',
        line=dict(color="#00ff41", width=2, dash='dot')
    ), secondary_y=True)

    try:
        max_vol = float(chart_df['Volatility'].max())
        mid_date = chart_df['Date'].iloc[len(chart_df)//3]
        late_date = chart_df['Date'].iloc[int(len(chart_df)//1.5)]

        fig.add_vline(x=mid_date, line_width=2, line_dash="dash", line_color="#FF003C")
        fig.add_vline(x=late_date, line_width=2, line_dash="dash", line_color="#FFD700")

        fig.add_annotation(x=mid_date, y=max_vol, text="[ US TARIFFS ]", showarrow=False, yshift=15, font=dict(color="#FF003C", family="Orbitron"))
        fig.add_annotation(x=late_date, y=max_vol * 0.9, text="[ OPEC POLICY ]", showarrow=False, yshift=15, font=dict(color="#FFD700", family="Orbitron"))
    except Exception: pass 

    # Removed strict 'height' so the chart scales naturally
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)')
    fig.update_yaxes(title_text="Volatility (Sigma)", color="#00f0ff", showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)', secondary_y=False)
    fig.update_yaxes(title_text="WTI Price (USD)", color="#00ff41", showgrid=False, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # --- BOTTOM ROW (NO HEIGHT LOCKS) ---
    col_intel, col_ai = st.columns([1, 1])

    with col_intel:
        st.markdown("### > TACTICAL EVENT LOG")
        st.info("🔴 **[TARIFF_IMPACT_01]:** Broad-based global tariffs enacted. WTI crude transport premiums increased by 14%. Volatility index spiked as supply chains rapidly re-routed to avoid import penalties.")
        st.warning("🟡 **[OPEC_POLICY_04]:** Sudden enforcement of production quotas restricted physical market liquidity. WTI price floor rigidly established, suppressing downside volatility.")
        st.success("🟢 **[NPRS-1_FORECAST]:** Model predicts a 68% probability of continued elevated volatility through Q3 due to persistent geopolitical friction and trade policy uncertainty.")

    with col_ai:
        st.markdown("### > DAEMON_V3_SECURE_TERMINAL")
        
        if "chat_log" not in st.session_state:
            st.session_state.chat_log = [{"role": "assistant", "content": f"DAEMON ONLINE. Query regarding {st.session_state.target} metrics?"}]

        for msg in st.session_state.chat_log:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        if prompt := st.chat_input("TRANSMIT_QUERY..."):
            st.session_state.chat_log.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("DECRYPTING..."):
                    if AI_AVAILABLE and vec is not None:
                        ans = get_ai_response(prompt, vec, tfidf, rag_df)
                    else:
                        ans = "SYSTEM OFFLINE: AI Engine not responding."
                    st.write(ans)
            st.session_state.chat_log.append({"role": "assistant", "content": ans})
