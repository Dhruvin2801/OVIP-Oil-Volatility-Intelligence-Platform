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
st.set_page_config(page_title="OVIP // Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #050810 !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        overflow-x: hidden !important; 
    }
    
    .block-container {
        padding-top: 1rem !important; 
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        margin-top: -40px !important; 
    }
    header {visibility: hidden;}

    .stApp::before {
        content: " "; display: block; position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }

    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .animated-panel {
        animation: slideUpFade 0.5s ease-out forwards;
        background: rgba(10, 15, 30, 0.8); 
        border: 1px solid rgba(0, 240, 255, 0.4); 
        border-radius: 4px; 
        padding: 15px;
        backdrop-filter: blur(5px);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.1);
        height: 100%; /* Ensure panels fill vertical space */
    }

    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f0ff !important;
        text-transform: uppercase;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        color: #00f0ff !important; 
        text-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    .stButton>button {
        background: rgba(0, 240, 255, 0.1) !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
        border-radius: 2px !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        transition: all 0.2s ease;
        padding: 5px 20px !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background: #00f0ff !important;
        color: #000 !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }
    
    .hud-box {
        background: rgba(0, 240, 255, 0.03); 
        border: 1px solid rgba(0, 240, 255, 0.2); 
        padding: 8px 15px; 
        border-radius: 2px;
        box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.02);
    }

    /* CSS Grid Layout for Country Dashboard */
    .grid-container {
        display: grid;
        grid-template-columns: 2fr 1fr;
        grid-gap: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PIPELINE
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
    return pd.DataFrame({'Date': dates, 'WTI': np.random.normal(75, 2, 150), 'Volatility': np.random.normal(0.15, 0.02, 150), 'Crisis_Prob': np.zeros(150), 'gpr': np.random.rand(150)*100})

df_main = load_data()

if 'rag_setup' not in st.session_state:
    if AI_AVAILABLE:
        try: st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df = setup_rag_vector_db(df_main)
        except: st.session_state.vec = None
    st.session_state.rag_setup = True

def hex_to_rgba(hex_code, alpha=0.1):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 6:
        return f'rgba({int(hex_code[0:2], 16)},{int(hex_code[2:4], 16)},{int(hex_code[4:6], 16)},{alpha})'
    return f'rgba(0,240,255,{alpha})'

# ==========================================
# 3. GLOBAL NODE DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE = '#00f0ff' 
C_WARN = '#ffd700'
C_DANG = '#ff003c'

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'info': 'Major importer. High sensitivity to Brent/Dubai spreads.', 'color': C_SAFE, 'mod': 0.95, 'figures': 'Refinery run rates at 102%. SPR expanding.', 'impact': 'Russian crude import discounts have shrunk.'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'info': 'Swing producer. Permian output highly elastic.', 'color': C_WARN, 'mod': 1.0, 'figures': 'Permian output steady at 6.1M bpd.', 'impact': 'Import tariffs enacted. SPR refill active.'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'info': 'Largest importer. Teapot refinery quotas dictate physical demand.', 'color': C_DANG, 'mod': 1.1, 'figures': 'Independent refinery utilization dropped to 58%.', 'impact': 'Tariff escalation reducing export demand.'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'info': 'Sanctioned exporter utilizing dark fleet logistics.', 'color': C_DANG, 'mod': 1.25, 'figures': 'Urals trading at $14 discount to Brent.', 'impact': 'Payment settlement delays increasing cargo float times.'},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'info': 'OPEC+ leader. ~3M bpd spare capacity.', 'color': C_WARN, 'mod': 1.0, 'figures': 'Holding ~3.2M bpd of spare capacity.', 'impact': 'Leaning toward resuming output increases.'},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'info': 'Sanctioned. Controls Strait of Hormuz chokepoint.', 'color': C_DANG, 'mod': 1.3, 'figures': 'Production recovering to 3.1M bpd.', 'impact': 'Transit harassment embedding geopolitical risk premium.'}
}

for k in ['UAE', 'VENEZUELA', 'BRAZIL', 'UK', 'NORWAY', 'NIGERIA', 'ANGOLA', 'LIBYA', 'IRAQ', 'KUWAIT', 'QATAR', 'CANADA', 'MEXICO', 'GERMANY', 'JAPAN', 'SOUTH KOREA', 'AUSTRALIA', 'ALGERIA', 'EGYPT', 'TURKEY', 'SOUTH AFRICA', 'SINGAPORE', 'INDONESIA', 'OMAN']:
    if k not in COUNTRIES:
        COUNTRIES[k] = {'lat': 0, 'lon': 0, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Generic tracking.', 'figures': 'Data aggregating...', 'impact': 'Monitoring macro headwinds.'}

COUNTRIES['UAE'].update({'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE})
COUNTRIES['VENEZUELA'].update({'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'color': C_DANG})
COUNTRIES['UK'].update({'lat': 55.37, 'lon': -3.43, 'color': C_WARN})

# ==========================================
# 4. THE AI MODAL (POP-UP)
# ==========================================
@st.dialog("OVIP AI Terminal")
def ai_terminal():
    st.markdown("<p style='color: #94a3b8; font-family: JetBrains Mono; font-size: 0.8em;'>> SECURE UPLINK ESTABLISHED.</p>", unsafe_allow_html=True)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "SYSTEM ONLINE. AWAITING QUERY."}]

    chat_box = st.container(height=300, border=False)
    with chat_box:
        for msg in st.session_state.chat_log:
            c = "#00f0ff" if msg['role'] == 'user' else "#e2e8f0"
            n = "USER" if msg['role'] == 'user' else "DAEMON"
            st.markdown(f"<span style='color:{c}; font-family: JetBrains Mono; font-size:0.9em;'><b>{n}:~$</b> {msg['content']}</span><br><br>", unsafe_allow_html=True)

    if prompt := st.chat_input("TRANSMIT..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.spinner("PROCESSING..."):
            if AI_AVAILABLE and st.session_state.vec is not None:
                ctx = f"Target: {st.session_state.target if st.session_state.target else 'Global'}. {prompt}"
                ans = get_ai_response(ctx, st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df)
            else: ans = "AI ENGINE OFFLINE."
        st.session_state.chat_log.append({"role": "assistant", "content": ans})
        st.rerun()

# ==========================================
# 5. MAIN VIEW: GLOBE OR DASHBOARD
# ==========================================
if st.session_state.target is None:
    # --- GLOBE VIEW WITH STAT BLOCKS ---
    c_head, c_btn = st.columns([8.5, 1.5]) 
    
    with c_head:
        st.markdown("<h2 style='margin-top: 5px; font-size: 1.8rem;'>GLOBAL_THREAT_MATRIX</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-family: JetBrains Mono; font-size: 12px; margin-top: -5px;'>> AWAITING_TARGET...</p>", unsafe_allow_html=True)
        
    with c_btn:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI"): ai_terminal()

    c_globe, c_stats = st.columns([7, 3])

    with c_globe:
        # Globe Plot
        lats = [v['lat'] for v in COUNTRIES.values()]
        lons = [v['lon'] for v in COUNTRIES.values()]
        names = list(COUNTRIES.keys())
        colors = [v['color'] for v in COUNTRIES.values()]
        
        fig_globe = go.Figure(go.Scattergeo(
            lon = lons, lat = lats, text = names, mode = 'markers+text',
            marker = dict(size=14, color=colors, line=dict(width=1, color='#050810')),
            textfont = dict(family="JetBrains Mono", size=12, color=colors),
            textposition = "top center"
        ))
        
        fig_globe.update_geos(
            projection_type="orthographic", showcoastlines=True, coastlinecolor="#1e3a8a", 
            showland=True, landcolor="#0f172a", showocean=True, oceancolor="#020617",         
            showcountries=True, countrycolor="#1e293b", bgcolor="rgba(0,0,0,0)",
            center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
        )
        
        fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        
        event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
        if event and "selection" in event and event["selection"]["points"]:
            st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
            st.rerun()

    with c_stats:
        # NPRS-1 ENGINE AND MACRO_ENVIRONMENT BLOCKS
        st.markdown("""
        <div class='animated-panel' style='margin-top: 20px; padding: 20px;'>
            <div style='color: #00f0ff; font-family: Orbitron; font-size: 14px; margin-bottom: 10px; letter-spacing: 1px; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 5px;'>[ NPRS-1 ENGINE ]</div>
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 2.0; margin-bottom: 30px;'>
                STATUS: <span style='color: #00f0ff;'>ONLINE</span><br>
                CONFIDENCE: <span style='color: #00f0ff;'>68.2% (UP_TREND)</span><br>
                NEXT_CYCLE: 12H:45M:03S
            </div>

            <div style='color: #00f0ff; font-family: Orbitron; font-size: 14px; margin-bottom: 10px; letter-spacing: 1px; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 5px;'>[ MACRO_ENVIRONMENT ]</div>
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 2.0;'>
                REGIME: <span style='color: #ffd700;'>MODERATE_RISK</span><br>
                CATALYST: TARIFF_POLICY_SHIFT<br>
                VOLATILITY: <span style='color: #00f0ff;'>ELEVATED (+1.4%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    # Header Row
    c_head, c_btn1, c_btn2 = st.columns([6, 2, 2])
    with c_head:
        st.markdown(f"<h2 style='color:{intel['color']}; margin-top: 5px; font-size: 1.8rem;'>NODE::{target}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-family: JetBrains Mono; font-size: 12px; margin-top: -5px;'>> UPLINK_ESTABLISHED</p>", unsafe_allow_html=True)
    with c_btn1:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("← GLOBE"): st.session_state.target = None; st.rerun()
    with c_btn2:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI"): ai_terminal()
        
    # Metrics Row
    st.markdown("<div class='animated-panel' style='margin-bottom: 15px; margin-top: 5px;'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WTI_PREMIUM (ADJ)", f"${(latest.get('WTI', 75) * mod):.2f}")
    m2.metric("LOCAL_VOL_SIGMA", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    m3.metric("GPR_INDEX", f"{(latest.get('gpr', 50) * mod):.1f}")
    
    m4.markdown(f"""
        <div style="display: flex; flex-direction: column;">
            <span style="color: #94a3b8; font-size: 14px; margin-bottom: 4px;">THREAT_LEVEL</span>
            <span style="color: {intel['color']}; font-family: 'Orbitron'; font-size: 1.8rem; font-weight: 700; text-shadow: 0 0 10px {intel['color']};">{intel['risk']}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Pure HTML/CSS Grid to Fix Empty Boxes
    st.markdown(f"""
    <div class="grid-container">
        <div class="animated-panel" style="height: 380px;" id="chart-container">
            <h4 style="color:#00f0ff;">VOLATILITY_IMPACT_MATRIX</h4>
        </div>
        <div class="animated-panel" style="height: 380px; overflow-y: auto;">
            <h4 style="color:{intel['color']}; border-bottom: 1px solid {hex_to_rgba(intel['color'], 0.3)}; padding-bottom: 10px;">LOCAL_EVENT_IMPACT</h4>
            <div style="font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-top: 15px;">
                <p style="color: #00f0ff; font-weight: bold;">[ MARKET_DATA ]</p>
                <p>{intel['figures']}</p>
                <div style="border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;"></div>
                <p style="color: {intel['color']}; font-weight: bold;">[ EVENT_IMPACT ]</p>
                <p>{intel['impact']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # We use Streamlit to render the chart inside a visually hidden container, then CSS positions it over the grid. 
    # This guarantees the layout works perfectly.
    chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hex_color = intel['color'].lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgba_fill = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.1)'
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Vol', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=rgba_fill), secondary_y=False)
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
    fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
    fig.update_yaxes(title_text="WTI", color="#475569", showgrid=False, secondary_y=True)
    
    # CSS trick to place the chart exactly over the chart-container grid box
    st.markdown("""
        <style>
            [data-testid="stPlotlyChart"] {
                margin-top: -360px;
                width: 65%;
            }
        </style>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
