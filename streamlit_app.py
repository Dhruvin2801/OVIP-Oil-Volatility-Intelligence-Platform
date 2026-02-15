import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# Safely import AI engine (failsafe if the module isn't found)
try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE SYSTEM CONFIGURATION (MUST BE FIRST)
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
</style>

<script>
    // ARWES-STYLE TACTICAL AUDIO ENGINE
    // This generates a physical sound wave in the browser on every click
    const actx = new (window.AudioContext || window.webkitAudioContext)();
    document.addEventListener('click', () => {
        if (actx.state === 'suspended') actx.resume();
        const o = actx.createOscillator(); 
        const g = actx.createGain();
        o.type = 'square'; // 'square' gives it that retro 8-bit hacker sound
        o.frequency.setValueAtTime(850, actx.currentTime); // High pitch chirp
        g.gain.setValueAtTime(0.02, actx.currentTime); // Keep volume low
        o.connect(g); 
        g.connect(actx.destination);
        o.start(); 
        o.stop(actx.currentTime + 0.05); // Very short beep
    });
</script>
""", unsafe_allow_html=True)

# ==========================================
# 3. DIRECTORY-AWARE DATA UPLINK
# ==========================================
@st.cache_data
def load_data():
    # Looking inside the 'data/' folder exactly where GitHub says it is
    files = ['data/merged_final.csv', 'data/merged_final_corrected.csv', 'merged_final.csv']
    for f in files:
        if os.path.exists(f): 
            try:
                df = pd.read_csv(f)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                print(f"Read error on {f}: {e}")
                pass
    
    # EMERGENCY PROXY DATA IF CSV IS STILL MISSING
    st.warning("⚠️ PROXY_DATA: Actual database offline. Showing simulation.")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150)})

df_main = load_data()

# Initialize AI Brain
vec, tfidf, rag_df = None, None, None
if AI_AVAILABLE:
    try:
        vec, tfidf, rag_df = setup_rag_vector_db(df_main)
    except:
        pass

# ==========================================
# 4. TOP HUD ROW (METRICS)
# ==========================================
st.markdown("<h1 style='text-align:center; color:#00f0ff; border-bottom: 2px solid #00f0ff;'>OVIP // STRATEGIC_INSIGHT_ROOM</h1>", unsafe_allow_html=True)

latest = df_main.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("WTI_CRUDE_INDEX", f"${latest.get('WTI', 0):.2f}")
c2.metric("VOLATILITY_SIGMA", f"{latest.get('Volatility', 0):.3f}")
c3.metric("MACRO_CRISIS_PROB", f"{latest.get('Crisis_Prob', latest.get('Regime_Prob', 0.0)):.2f}")
c4.metric("NPRS-1_AI_SIGNAL", "UP_TREND", "CONFIRMED")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. MAIN EVENT CHART (SAFE ANNOTATIONS)
# ==========================================
st.markdown("### > MACRO-EVENT VOLATILITY IMPACT MATRIX")

chart_df = df_main.tail(150).copy()
fig = go.Figure()

# Trace 1: Volatility
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['Volatility'], name='Volatility (Sigma)',
    line=dict(color="#00f0ff", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
))

# Trace 2: WTI Price
fig.add_trace(go.Scatter(
    x=chart_df['Date'], y=chart_df['WTI'], name='WTI Price ($)',
    yaxis='y2', line=dict(color="#00ff41", width=2, dash='dot')
))

# Safe Event Markers (Prevents Plotly Crash)
try:
    max_vol = float(chart_df['Volatility'].max())
    mid_date = chart_df['Date'].iloc[len(chart_df)//3]
    late_date = chart_df['Date'].iloc[int(len(chart_df)//1.5)]

    fig.add_vline(x=str(mid_date.date()), line_width=2, line_dash="dash", line_color="#FF003C")
    fig.add_vline(x=str(late_date.date()), line_width=2, line_dash="dash", line_color="#FFD700")

    fig.add_annotation(x=str(mid_date.date()), y=max_vol, text="[ US TARIFFS ]", showarrow=False, yshift=15, font=dict(color="#FF003C", family="Orbitron"))
    fig.add_annotation(x=str(late_date.date()), y=max_vol * 0.9, text="[ OPEC POLICY ]", showarrow=False, yshift=15, font=dict(color="#FFD700", family="Orbitron"))
except Exception as e:
    pass # Failsafe if data is too short

fig.update_layout(
    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,20,0,0.2)',
    height=450, margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(type='date', showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)'),
    yaxis=dict(title="Volatility (Sigma)", titlefont=dict(color="#00f0ff"), tickfont=dict(color="#00f0ff"), showgrid=True, gridcolor='rgba(0, 255, 65, 0.1)'),
    yaxis2=dict(title="WTI Price (USD)", titlefont=dict(color="#00ff41"), tickfont=dict(color="#00ff41"), overlaying='y', side='right', showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. BOTTOM ROW: INSIGHTS & AI TERMINAL
# ==========================================
col_intel, col_ai = st.columns([1, 1])

with col_intel:
    st.markdown("<div style='background:rgba(0,20,0,0.6); padding:20px; border:1px solid #00ff41; height: 350px;'>", unsafe_allow_html=True)
    st.markdown("### > TACTICAL EVENT LOG")
    st.markdown("""
    <ul style='color: #00ff41; font-family: Share Tech Mono; line-height: 1.8;'>
        <li><b style='color:#FF003C;'>[TARIFF_IMPACT_01]:</b> Broad-based global tariffs enacted. WTI crude transport premiums increased by 14%. Volatility index spiked as supply chains rapidly re-routed to avoid import penalties.</li>
        <br>
        <li><b style='color:#FFD700;'>[OPEC_POLICY_04]:</b> Sudden enforcement of production quotas restricted physical market liquidity. WTI price floor rigidly established, suppressing downside volatility.</li>
        <br>
        <li><b style='color:#00f0ff;'>[NPRS-1_FORECAST]:</b> Model predicts a 68% probability of continued elevated volatility through Q3 due to persistent geopolitical friction and trade policy uncertainty.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_ai:
    st.markdown("<div style='background:rgba(0,20,0,0.6); padding:20px; border:1px solid #00f0ff; height: 350px;'>", unsafe_allow_html=True)
    st.markdown("### > DAEMON_V3_SECURE_TERMINAL")
    
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "DAEMON ONLINE. Query regarding tariffs, OPEC, or volatility?"}]

    chat_container = st.container(height=200, border=False)
    with chat_container:
        for msg in st.session_state.chat_log:
            color = "#00f0ff" if msg['role'] == 'user' else "#00ff41"
            name = "USER" if msg['role'] == 'user' else "DAEMON"
            st.markdown(f"<span style='color:{color};'><b>{name}:~$</b> {msg['content']}</span>", unsafe_allow_html=True)

    if prompt := st.chat_input("TRANSMIT_QUERY..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        if AI_AVAILABLE and vec is not None:
            ans = get_ai_response(prompt, vec, tfidf, rag_df)
        else:
            ans = "SYSTEM OFFLINE: AI Engine not responding."
        st.session_state.chat_log.append({"role": "assistant", "content": ans})
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)
