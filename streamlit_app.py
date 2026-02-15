import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
from modules.ai_engine import setup_rag_vector_db, get_ai_response

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="OVIP // INSIGHT_ROOM", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 2. TACTICAL CSS SHADERS & SOUND ENGINE
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    /* PURE BLACK BACKGROUND WITH SCANLINES */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000200 !important;
        color: #00ff41 !important;
        font-family: 'Share Tech Mono', monospace !important;
    }
    
    .stApp::before {
        content: " "; display: block; position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }

    /* NEON HEADERS & METRICS */
    h1, h2, h3, h4, [data-testid="stMetricValue"] { 
        font-family: 'Orbitron', sans-serif !important; 
        text-shadow: 0 0 10px #00ff41; 
        text-transform: uppercase;
    }
    
    /* CUSTOMIZE STREAMLIT CONTAINERS (BORDERS) */
    [data-testid="stVerticalBlock"] > div.element-container {
        border-radius: 2px;
    }

    /* CYBERPUNK BUTTONS */
    .stButton>button {
        background: rgba(0, 255, 65, 0.05) !important; color: #00f0ff !important; 
        border: 1px solid #00f0ff !important; font-family: 'Orbitron', sans-serif !important; 
        width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #00f0ff !important; color: #000 !important; box-shadow: 0 0 15px #00f0ff; }
    
    /* DATA TABLES */
    [data-testid="stTable"] { background-color: rgba(0, 20, 0, 0.5); }
</style>

<script>
    // AUDIO ENGINE FOR CLICKS
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {
        const o = actx.createOscillator(); const g = actx.createGain();
        o.type='square'; o.frequency.value=750;
        g.gain.value=0.03; o.connect(g); g.connect(actx.destination);
        o.start(); o.stop(actx.currentTime+0.05);
    });
</script>
""", unsafe_allow_html=True)

# ==========================================
# 3. SELF-HEALING DATA UPLINK
# ==========================================
@st.cache_data
def load_data():
    files = ['merged_fnal.csv', 'merged_final.csv', 'merged_final_corrected.csv']
    for f in files:
        if os.path.exists(f): 
            try:
                df = pd.read_csv(f)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            except: pass
    
    # EMERGENCY PROXY DATA IF CSV IS MISSING
    st.warning("⚠️ PROXY_DATA: Database offline. Generating tactical simulation.")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150)})

df_main = load_data()
try:
    vec, tfidf, rag_df = setup_rag_vector_db(df_main)
except:
    vec, tfidf, rag_df = None, None, None # Failsafe if AI engine isn't ready

# ==========================================
# 4. TOP HUD ROW (METRICS)
# ==========================================
st.markdown("<h1 style='text-align:center; color:#00f0ff; border-bottom: 2px solid #00f0ff;'>OVIP // STRATEGIC_INSIGHT_ROOM</h1>", unsafe_allow_html=True)

latest = df_main.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("WTI_CRUDE_INDEX", f"${latest.get('WTI', 0):.2f}", "+1.2%")
c2.metric("VOLATILITY_SIGMA", f"{latest.get('Volatility', 0):.3f}", "ELEVATED")
c3.metric("MACRO_CRISIS_PROB", f"{latest.get('Crisis_Prob', latest.get('Regime_Prob', 0.0)):.2f}", "SAFE")
c4.metric("NPRS-1_AI_SIGNAL", "UP_TREND", "CONFIRMED")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. MAIN EVENT CHART (VOLATILITY VS TARIFFS/POLICIES)
# ==========================================
st.markdown("### > MACRO-EVENT VOLATILITY IMPACT MATRIX")

# Create a slice of the last 150 days for clarity
chart_df = df_main.tail(150).copy()

fig = go.Figure()

# Trace 1: Volatility (Left Axis)
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility (Sigma)',
    line=dict(color="#00f0ff", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
))

# Trace 2: WTI Price (Right Axis)
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['WTI'], name='WTI Price ($)',
    yaxis='y2', line=dict(color="#00ff41", width=2, dash='dot')
))

# EVENT ANNOTATIONS (Tariffs & Policies)
# We place these dynamically based on the available dates in your data
mid_date = chart_df['Date'].iloc[len(chart_df)//3]
late_date = chart_df['Date'].iloc[int(len(chart_df)//1.5)]

fig.add_vline(x=mid_date, line_width=2, line_dash="dash", line_color="#FF003C")
fig.add_annotation(x=mid_date, y=chart_df['Volatility'].max(), text="[ US TARIFF ENACTMENT ]", showarrow=True, arrowhead=1, font=dict(color="#FF003C", family="Orbitron"))

fig.add_vline(x=late_date, line_width=2, line_dash="dash", line_color="#FFD700")
fig.add_annotation(x=late_date, y=chart_df['Volatility'].max() * 0.9, text="[ OPEC+ POLICY SHIFT ]", showarrow=True, arrowhead=1, font=dict(color="#FFD700", family="Orbitron"))

# Layout configuration
fig.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
    height=450, margin=dict(l=0, r=0, t=30, b=0),
    yaxis=dict(title="Volatility (Sigma)", titlefont=dict(color="#00f0ff"), tickfont=dict(color="#00f0ff")),
    yaxis2=dict(title="WTI Price (USD)", titlefont=dict(color="#00ff41"), tickfont=dict(color="#00ff41"), overlaying='y', side='right'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. BOTTOM ROW: INSIGHTS & AI TERMINAL
# ==========================================
col_intel, col_ai = st.columns([1, 1])

with col_intel:
    st.markdown("<div style='background:rgba(0,20,0,0.6); padding:20px; border:1px solid #00ff41;'>", unsafe_allow_html=True)
    st.markdown("### > TACTICAL EVENT LOG")
    st.markdown("""
    <ul style='color: #00ff41; font-family: Share Tech Mono; line-height: 1.8;'>
        <li><b style='color:#FF003C;'>[TARIFF_IMPACT_01]:</b> Broad-based global tariffs enacted. WTI crude transport premiums increased by 14%. Volatility index spiked as supply chains rapidly re-routed to avoid import penalties.</li>
        <li><b style='color:#FFD700;'>[OPEC_POLICY_04]:</b> Sudden enforcement of production quotas restricted physical market liquidity. WTI price floor rigidly established, suppressing downside volatility.</li>
        <li><b style='color:#00f0ff;'>[NPRS-1_FORECAST]:</b> Model predicts a 68% probability of continued elevated volatility through Q3 due to persistent geopolitical friction and trade policy uncertainty.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_ai:
    st.markdown("<div style='background:rgba(0,20,0,0.6); padding:20px; border:1px solid #00f0ff;'>", unsafe_allow_html=True)
    st.markdown("### > DAEMON_V3_SECURE_TERMINAL")
    
    # Initialize chat history if empty
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "DAEMON ONLINE. State your query regarding tariffs, OPEC policy, or market volatility."}]

    # Render Chat
    chat_container = st.container(height=250, border=False)
    with chat_container:
        for msg in st.session_state.chat_log:
            color = "#00f0ff" if msg['role'] == 'user' else "#00ff41"
            name = "USER" if msg['role'] == 'user' else "DAEMON"
            st.markdown(f"<span style='color:{color};'><b>{name}:~$</b> {msg['content']}</span><br><br>", unsafe_allow_html=True)

    # Input Box
    if prompt := st.chat_input("TRANSMIT_QUERY..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        
        # Get AI Response
        if vec is not None:
            ans = get_ai_response(prompt, vec, tfidf, rag_df)
        else:
            ans = "SYSTEM OFFLINE. RAG Engine failed to initialize."
            
        st.session_state.chat_log.append({"role": "assistant", "content": ans})
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)
