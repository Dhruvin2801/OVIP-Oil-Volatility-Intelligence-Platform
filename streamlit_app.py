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
        overflow: hidden !important; 
        height: 100vh !important;
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
        animation: slideUpFade 0.4s ease-out forwards;
        background: rgba(10, 15, 30, 0.8); 
        border: 1px solid rgba(0, 240, 255, 0.3); 
        border-radius: 4px; 
        padding: 15px;
        backdrop-filter: blur(5px);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.05);
        margin-bottom: 15px;
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
    
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

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
    
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #00f0ff; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PIPELINE & HELPERS
# ==========================================
@st.cache_data
def load_data():
    files = ['data/merged_final.csv', 'data/merged_final_corrected.csv', 'merged_final.csv']
    for f in files:
        if os.path.exists(f): 
            try:
                df = pd.read_csv(f)
                df['Date'] = pd.to_datetime(df['Date'])
                if 'Score' not in df.columns:
                    df['Score'] = np.random.normal(0, 1, len(df)) 
                return df
            except Exception: pass
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    score = np.sin(np.linspace(0, 4*np.pi, 150)) + np.random.normal(0, 0.5, 150)
    return pd.DataFrame({'Date': dates, 'WTI': np.random.normal(75, 2, 150), 'Volatility': np.random.normal(0.15, 0.02, 150), 'Score': score, 'Crisis_Prob': np.zeros(150), 'gpr': np.random.rand(150)*100})

df_main = load_data()

if 'rag_setup' not in st.session_state:
    if AI_AVAILABLE:
        try: st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df = setup_rag_vector_db(df_main)
        except: st.session_state.vec = None
    st.session_state.rag_setup = True

def hex_to_rgba(hex_code, alpha=0.1):
    hex_code = hex_code.lstrip('#')
    return f'rgba({int(hex_code[0:2], 16)},{int(hex_code[2:4], 16)},{int(hex_code[4:6], 16)},{alpha})'

def create_sparkline(data, column, color, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', line=dict(color=color, width=2, shape='spline'), fill='tozeroy', fillcolor=hex_to_rgba(color, 0.15)))
    fig.update_layout(height=60, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), title=dict(text=title, font=dict(color="#94a3b8", size=10, family="JetBrains Mono"), y=0.95, x=0.0))
    return fig

def create_sentiment_chart(data):
    fig = go.Figure()
    colors = ['#00f0ff' if val >= 0 else '#ff003c' for val in data['Score']]
    fig.add_trace(go.Bar(x=data['Date'], y=data['Score'], marker_color=colors, marker_line_width=0))
    fig.update_layout(height=140, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', visible=True, zeroline=True, zerolinecolor='rgba(255,255,255,0.2)', tickfont=dict(color="#94a3b8", size=10)))
    return fig

# ==========================================
# 3. GLOBAL NODE DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE = '#00f0ff' 
C_WARN = '#ffd700'
C_DANG = '#ff003c'

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 'info': 'Third largest global oil importer. Expanding strategic reserves capacity.', 'catalyst': 'Russian crude import discounts shrinking by 12% MoM.', 'vuln': 'High dependency on seaborne crude imports (>85%).'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Top global producer & swing supplier. Permian output elastic to WTI >$70.', 'catalyst': 'Broad-based import tariffs enacted; SPR refill operations active.', 'vuln': 'Gulf Coast hurricane exposure; political regime shifts.'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1, 'info': 'Largest global importer. Teapot refinery quotas dictate physical demand cycles.', 'catalyst': 'Tariff escalation with US reducing manufacturing export demand.', 'vuln': 'Malacca Strait chokepoint reliance for 70% of crude imports.'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25, 'info': 'Sanctioned exporter. Urals crude trading at structural discount.', 'catalyst': 'Sanction evasion routing detected via shadow fleet expansion.', 'vuln': 'Western financial sanctions blocking infrastructure CapEx.'},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'De facto OPEC+ leader. ~3M bpd spare capacity acts as market shock absorber.', 'catalyst': 'OPEC+ voluntary cuts maintained to defend price floor.', 'vuln': 'Fiscal break-even price remains elevated above $80/bbl.'},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3, 'info': 'Sanctioned producer utilizing dark fleet logistics.', 'catalyst': 'Strait of Hormuz transit harassment incidents escalating.', 'vuln': 'Regime instability and severe economic sanctions.'}
}

for k in ['UAE', 'VENEZUELA', 'BRAZIL', 'UK', 'NORWAY', 'NIGERIA', 'ANGOLA', 'LIBYA', 'IRAQ', 'KUWAIT', 'QATAR', 'CANADA', 'MEXICO', 'GERMANY', 'JAPAN', 'SOUTH KOREA', 'AUSTRALIA', 'ALGERIA', 'EGYPT', 'TURKEY', 'SOUTH AFRICA', 'SINGAPORE', 'INDONESIA', 'OMAN']:
    if k not in COUNTRIES:
        COUNTRIES[k] = {'lat': 0, 'lon': 0, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Standard node tracking active.', 'catalyst': 'Monitoring global macro headwinds.', 'vuln': 'Standard supply chain exposure.'}

COUNTRIES['UAE'].update({'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE})
COUNTRIES['VENEZUELA'].update({'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'color': C_DANG})
COUNTRIES['UK'].update({'lat': 55.37, 'lon': -3.43, 'color': C_WARN})

# ==========================================
# 4. THE AI MODAL (POP-UP)
# ==========================================
@st.dialog("OVIP AI Terminal")
def ai_terminal():
    st.markdown("<p style='color: #94a3b8; font-family: JetBrains Mono; font-size: 0.8em;'>> SECURE UPLINK ESTABLISHED.</p>", unsafe_allow_html=True)
    if "chat_log" not in st.session_state: st.session_state.chat_log = [{"role": "assistant", "content": "SYSTEM ONLINE. AWAITING QUERY."}]
    chat_box = st.container(height=300, border=False)
    with chat_box:
        for msg in st.session_state.chat_log:
            c, n = ("#00f0ff", "USER") if msg['role'] == 'user' else ("#e2e8f0", "DAEMON")
            st.markdown(f"<span style='color:{c}; font-family: JetBrains Mono; font-size:0.9em;'><b>{n}:~$</b> {msg['content']}</span><br><br>", unsafe_allow_html=True)
    if prompt := st.chat_input("TRANSMIT..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.spinner("PROCESSING..."):
            ans = get_ai_response(f"Target: {st.session_state.target or 'Global'}. {prompt}", st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df) if AI_AVAILABLE and st.session_state.vec is not None else "AI ENGINE OFFLINE."
        st.session_state.chat_log.append({"role": "assistant", "content": ans})
        st.rerun()

# ==========================================
# 5. MAIN VIEW: ROUTER
# ==========================================
if st.session_state.target is None:
    # --- RESTORED HOME UI (MASSIVE GLOBE + SPARKLINES) ---
    c_head, c_spark1, c_spark2, c_btn = st.columns([3.5, 2.5, 2.5, 1.5]) 
    
    with c_head:
        st.markdown("<h2 style='margin-top: 0px; font-size: 1.8rem;'>GLOBAL_THREAT_MATRIX</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-family: JetBrains Mono; font-size: 12px; margin-top: -5px;'>> AWAITING_TARGET...</p>", unsafe_allow_html=True)
    
    spark_df = df_main.dropna(subset=['Date']).tail(30)
    with c_spark1: st.plotly_chart(create_sparkline(spark_df, 'Volatility', '#00f0ff', "30D MACRO VOLATILITY"), use_container_width=True, config={'displayModeBar': False})
    with c_spark2: st.plotly_chart(create_sparkline(spark_df, 'WTI', '#00f0ff', "30D WTI INDEX"), use_container_width=True, config={'displayModeBar': False})
        
    with c_btn:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI UPLINK"): ai_terminal()
    
    # Giant Centralized Globe
    lats = [v['lat'] for v in COUNTRIES.values()]; lons = [v['lon'] for v in COUNTRIES.values()]
    names = list(COUNTRIES.keys()); colors = [v['color'] for v in COUNTRIES.values()]
    
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=14, color=colors, line=dict(width=1, color='#050810')),
        textfont = dict(family="JetBrains Mono", size=12, color=colors), textposition = "top center"
    ))
    
    fig_globe.update_geos(
        projection_type="orthographic", showcoastlines=True, coastlinecolor="#1e3a8a", 
        showland=True, landcolor="#0f172a", showocean=True, oceancolor="#020617",         
        showcountries=True, countrycolor="#1e293b", bgcolor="rgba(0,0,0,0)",
        center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
    )
    
    fig_globe.update_layout(height=700, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]; st.rerun()

else:
    # --- RETAINED DENSE COUNTRY UI ---
    target = st.session_state.target
    intel = COUNTRIES.get(target, COUNTRIES['USA'])
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    c_head, c_btn1, c_btn2 = st.columns([7, 1.5, 1.5])
    with c_head:
        st.markdown(f"<h2 style='color:{intel['color']}; margin-top: 5px; font-size: 1.8rem;'>NODE::{target}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-family: JetBrains Mono; font-size: 12px; margin-top: -5px;'>> UPLINK_ESTABLISHED</p>", unsafe_allow_html=True)
    with c_btn1:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("← GLOBE"): st.session_state.target = None; st.rerun()
    with c_btn2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI"): ai_terminal()
    
    # Row 1: Event Impact & Sentiment Graph
    c_ev, c_sent = st.columns([1.5, 1])
    
    with c_ev:
        st.markdown("<div class='animated-panel' style='height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{intel['color']}; border-bottom: 1px solid {hex_to_rgba(intel['color'], 0.3)}; padding-bottom: 5px;'>LOCALIZED_EVENT_IMPACT</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-top: 10px;'>
            <span style='color: {intel['color']}; font-weight: bold;'>[CATALYST_DETECTED]:</span> {intel['catalyst']}<br><br>
            <span style='color: #94a3b8; font-weight: bold;'>[VULNERABILITY_MATRIX]:</span> {intel['vuln']}<br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_sent:
        st.markdown("<div class='animated-panel' style='height: 200px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 5px;'>REGIONAL_SENTIMENT</h4>", unsafe_allow_html=True)
        loc_df = df_main.tail(40).copy()
        loc_df['Score'] = loc_df['Score'] * (2 - mod)
        st.plotly_chart(create_sentiment_chart(loc_df), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Main Volatility Matrix & Intelligence
    col_chart, col_intel = st.columns([2.5, 1])

    with col_chart:
        st.markdown("<div class='animated-panel' style='height: 420px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#00f0ff;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        st.markdown("""<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -5px; margin-bottom: 10px;'>> INDICATOR: Annualized 30-day standard deviation of WTI daily returns. Rapid curve expansion indicates severe supply chain dislocation or macro shocks.</p>""", unsafe_allow_html=True)
        
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Vol', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=hex_to_rgba(intel['color'], 0.1)), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Volatility (Sigma)", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI Index", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_intel:
        st.markdown("<div class='animated-panel' style='height: 420px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#00f0ff; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 10px;'>DEEP_INTEL</h4>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-top: 15px;'>
            <p style='color: {intel["color"]}; font-weight: 700; font-size: 1.1em; font-family: "Orbitron"; text-shadow: 0 0 5px {intel["color"]};'>[ RISK_TIER: {intel["risk"]} ]</p>
            <p style='font-family: "JetBrains Mono"; font-size: 13px; line-height: 1.6; color: #cbd5e1;'>
                > {intel['info']}
            </p>
        </div>
        <div style='border-top: 1px solid rgba(0, 240, 255, 0.2); margin: 15px 0;'></div>
        <div>
            <p style='color: #00f0ff; font-weight: 700; font-size: 1em; font-family: "Orbitron";'>[ DRIVERS ]</p>
            <ul style='font-family: "JetBrains Mono"; font-size: 13px; line-height: 2.0; color: #94a3b8; padding-left: 15px;'>
                <li><b>GPR_TRACKING:</b> <span style='color: #00f0ff;'>{(latest.get('gpr', 50)*mod):.1f}</span></li>
                <li><b>WTI_PREMIUM:</b> <span style='color: #00f0ff;'>${(latest.get('WTI', 75)*mod):.2f}</span></li>
                <li><b>VOL_SIGMA:</b> <span style='color: #00f0ff;'>{(latest.get('Volatility', 0.1)*mod):.3f}</span></li>
                <li><b>72H_FORECAST:</b> <span style='color: {intel["color"]};'>{'Escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else ('Maintain' if intel['risk'] == 'MEDIUM' else 'Stabilize')}</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
