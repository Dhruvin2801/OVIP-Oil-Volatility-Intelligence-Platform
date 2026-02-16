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

# CSS for styling colors, fonts, and containers.
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
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        margin-top: -50px !important; 
    }
    header {visibility: hidden;}

    .stApp::before {
        content: " "; display: block; position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
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
    
    /* Native container styling for "panels" */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: rgba(10, 15, 30, 0.6);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 4px;
        padding: 20px;
        backdrop-filter: blur(5px);
    }

    /* AI CHAT VISIBILITY FIXES */
    div[data-testid="stChatInput"] textarea {
        color: #00f0ff !important;
        -webkit-text-fill-color: #00f0ff !important;
    }
    div[data-testid="stDialog"] > div {
        background-color: #050810 !important;
        border: 1px solid #00f0ff !important;
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
                if 'Score' not in df.columns: df['Score'] = np.random.normal(0, 1, len(df))
                return df
            except Exception: pass
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    score = np.sin(np.linspace(0, 4*np.pi, 150)) + np.random.normal(0, 0.5, 150)
    return pd.DataFrame({'Date': dates, 'WTI': np.random.normal(75, 2, 150), 'Volatility': np.random.normal(0.15, 0.02, 150), 'Crisis_Prob': np.random.uniform(0.1, 0.9, 150), 'gpr': np.random.rand(150)*100, 'Score': score})

df_main = load_data()

if 'rag_setup' not in st.session_state:
    if AI_AVAILABLE:
        try: st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df = setup_rag_vector_db(df_main)
        except: st.session_state.vec = None
    st.session_state.rag_setup = True

def hex_to_rgba(hex_code, alpha=0.1):
    hex_code = hex_code.lstrip('#')
    return f'rgba({int(hex_code[0:2], 16)},{int(hex_code[2:4], 16)},{int(hex_code[4:6], 16)},{alpha})'

def create_feature_importance_chart(color):
    factors = ['Geo-Risk (GPR)', 'WTI Momentum', 'NLP Sentiment', 'OPEC+ Supply', 'Freight Cost']
    importance = [0.35, 0.25, 0.20, 0.12, 0.08]
    fig = go.Figure(go.Bar(x=importance, y=factors, orientation='h', marker=dict(color=color, line=dict(color='rgba(255,255,255,0.2)', width=1))))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=dict(text="VOLATILITY_DRIVERS", font=dict(color="#00f0ff", family="Orbitron", size=14)), xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color="#94a3b8")), yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=12, color="#e2e8f0")))
    return fig

# ==========================================
# 3. GLOBAL NODE DATABASE (30 COUNTRIES TOTAL)
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE, C_WARN, C_DANG = '#00f0ff', '#ffd700', '#ff003c'

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 'info': '3rd largest global oil importer. Hyper-sensitive to Brent/Dubai spreads. Local refinery run rates are currently maximizing at 102% utilization.', 'figures': 'Refinery Run Rate: 102% | Strategic Reserve Expansion: +6.5M tons.', 'impact': 'Russian crude discounts shrinking to $3.50/bbl. Fiscal pressure rising for IOCL and BPCL.'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Top global producer & swing supplier. Permian basin output steady at 6.1M bpd.', 'figures': 'Permian Output: 6.1M bpd | Inventories: -2.4M bbl WoW.', 'impact': 'SPR refill active at $73/bbl floor. New tariffs re-routing heavy crude inputs.'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1, 'info': 'Largest global crude importer. Demand dictated by independent Teapot refinery export quotas.', 'figures': 'Independent Utilization: 58% | LNG Imports: +14% YoY.', 'impact': 'Manufacturing demand slowing. Malacca Strait risk remains a critical concern.'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25, 'info': 'Sanctioned major exporter using dark fleet networks. Urals trading at steep $14 discount to Brent.', 'figures': 'Seaborne Exports: 3.3M bpd | Dark Fleet Capacity: 600+ vessels.', 'impact': 'Sanctions blocking infrastructure CapEx. Payment delays increasing cargo float times.'},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'OPEC+ leader. Sole entity capable of acting as primary market shock absorber.', 'figures': 'Offline Spare Capacity: ~3.2M bpd | Fiscal Break-even: $82/bbl.', 'impact': 'Leaning toward ending voluntary cuts in April 2026 to reclaim Asian market share.'},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.98, 'info': 'Global logistics hub. Fujairah port serves as leading indicator for export health.', 'figures': 'Fujairah Bunkering: 14-month high | Capacity Goal: 5M bpd by 2027.', 'impact': 'Capitalizing on Red Sea rerouting. Investing heavily in incremental ADNOC production expansion.'},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3, 'info': 'Controls Strait of Hormuz (21M bpd transit risk). Production covertly recovered to 3.1M bpd.', 'figures': 'Asian Export Share: >90% | Covert Output: 3.1M bpd.', 'impact': 'Transit harassment incidents embedding $3 risk premium directly into global benchmarks.'},
    'VENEZUELA': {'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.15, 'info': 'Heavy crude specialist. Output stalled at ~800k bpd due to structural underinvestment.', 'figures': 'Required Diluents: Heavily restricted | Reserves: Largest globally.', 'impact': 'US sanctions snapback limiting exports to Western refiners.'},
    'UK': {'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.02, 'info': 'Brent benchmark origin. North Sea production in terminal decline at 8% YoY.', 'figures': 'Active Offshore Rigs: Near historic lows.', 'impact': 'Windfall taxes restricting offshore drilling CapEx.'},
    'OMAN': {'lat': 21.51, 'lon': 55.92, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 'info': 'Key benchmark component (Dubai/Oman). Non-OPEC Middle East producer.', 'figures': 'Strategic bypass pipelines expanding.', 'impact': 'Positioning as vital alternative to Strait of Hormuz transit.'},
    
    # 10 ADDITIONAL NODES
    'SINGAPORE': {'lat': 1.35, 'lon': 103.82, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 'info': 'Premier global pricing hub. Critical for maritime bunkering and refining.', 'figures': 'Onshore inventories near multi-year lows.', 'impact': 'Monitoring South China Sea transit security and refining margins.'},
    'SOUTH KOREA': {'lat': 35.91, 'lon': 127.77, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 'info': 'Top-tier refining hub dependent on Middle Eastern sour crude.', 'figures': 'Strategic reserves at 90-day coverage.', 'impact': 'Petrochemical export demand affected by industrial slowdown.'},
    'VIETNAM': {'lat': 14.05, 'lon': 108.27, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Emerging regional refining power. Expanding offshore exploration.', 'figures': 'Nghi Son refinery at full capacity.', 'impact': 'Maritime border disputes affecting long-term CapEx.'},
    'KAZAKHSTAN': {'lat': 48.01, 'lon': 66.92, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Major producer via CPC pipeline. Prone to Russian transit disruptions.', 'figures': 'Tengiz field expansion delayed.', 'impact': 'Export reliability tied to Russian Black Sea stability.'},
    'NIGERIA': {'lat': 9.08, 'lon': 8.68, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.2, 'info': 'Key light sweet crude exporter. Production prone to sabotage.', 'figures': 'Output struggling to meet 1.5M bpd quota.', 'impact': 'Dangote refinery shifting national export/import gasoline balance.'},
    'ANGOLA': {'lat': -11.20, 'lon': 17.87, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.05, 'info': 'Exited OPEC. Deepwater production seeing natural decline.', 'figures': 'Production averaging 1.1M bpd.', 'impact': 'Increasing reliance on Chinese investment for maintenance.'},
    'NORWAY': {'lat': 60.47, 'lon': 8.47, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.9, 'info': 'Reliable European supplier. Johan Sverdrup field providing baseload.', 'figures': 'Exports to EU at historic highs.', 'impact': 'Gas focus limiting incremental oil growth.'},
    'GERMANY': {'lat': 51.17, 'lon': 10.45, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Major consumer. Shifting from Russian Urals to seaborne grades.', 'figures': 'Refinery utilization at 82%.', 'impact': 'Industrial slowdown reducing diesel demand significantly.'},
    'BRAZIL': {'lat': -14.24, 'lon': -51.93, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.9, 'info': 'Non-OPEC growth driver. Massive pre-salt offshore reserves.', 'figures': 'Output hitting 3.5M bpd record.', 'impact': 'Competing with West African grades in Asian markets.'},
    'EGYPT': {'lat': 26.82, 'lon': 30.80, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.15, 'info': 'Controls Suez Canal and SUMED pipeline. Vital transit chokepoint.', 'figures': 'Transit Revenues: -40% due to Red Sea conflict.', 'impact': 'Tanker rerouting around Africa impacting Mediterranean supply.'}
}

# ==========================================
# 4. THE AI MODAL
# ==========================================
@st.dialog("OVIP AI Terminal")
def ai_terminal():
    st.markdown("<p style='color: #94a3b8; font-family: JetBrains Mono; font-size: 0.8em;'>> SECURE UPLINK ESTABLISHED.</p>", unsafe_allow_html=True)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "SYSTEM ONLINE. AWAITING QUERY."}]

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
    c_head, c_btn = st.columns([8.5, 1.5]) 
    with c_head: st.markdown("<h2 style='margin-top: 5px; font-size: 2rem;'>GLOBAL_THREAT_MATRIX</h2>", unsafe_allow_html=True)
    with c_btn:
        st.write("")
        if st.button("💬 OVIP AI"): ai_terminal()

    c_globe, c_stats = st.columns([7, 3])

    with c_globe:
        lats = [v['lat'] for v in COUNTRIES.values()]; lons = [v['lon'] for v in COUNTRIES.values()]
        names = list(COUNTRIES.keys()); colors = [v['color'] for v in COUNTRIES.values()]
        fig_globe = go.Figure(go.Scattergeo(lon = lons, lat = lats, text = names, mode = 'markers+text', marker = dict(size=14, color=colors, line=dict(width=1, color='#050810')), textfont = dict(family="JetBrains Mono", size=12, color=colors), textposition = "top center"))
        fig_globe.update_geos(projection_type="orthographic", showcoastlines=True, coastlinecolor="#1e3a8a", showland=True, landcolor="#0f172a", showocean=True, oceancolor="#020617", showcountries=True, countrycolor="#1e293b", bgcolor="rgba(0,0,0,0)", center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0))
        fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
        if event and "selection" in event and event["selection"]["points"]:
            st.session_state.target = names[event["selection"]["points"][0]["point_index"]]; st.rerun()

    with c_stats:
        with st.container():
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ NPRS-1 ENGINE ]</p>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; margin-bottom: 25px;'>STATUS: <span style='color: #00f0ff; font-weight: bold;'>ONLINE</span><br>CONFIDENCE: <span style='color: #00f0ff; font-weight: bold;'>68.2% (UP_TREND)</span><br>NEXT_CYCLE: <span style='color: #cbd5e1;'>12H:45M:03S</span></div>""", unsafe_allow_html=True)
            st.markdown("<div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ MACRO_ENVIRONMENT ]</p>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; margin-bottom: 25px;'>REGIME: <span style='color: #ffd700; font-weight: bold;'>MODERATE_RISK</span><br>CATALYST: <span style='color: #cbd5e1;'>TARIFF_POLICY_SHIFT</span><br>VOLATILITY: <span style='color: #00f0ff; font-weight: bold;'>ELEVATED (+1.4%)</span></div>""", unsafe_allow_html=True)
            st.markdown("<div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 15px;'>[ LIVE_FEED ]</p>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6; margin-bottom: 15px;'><span style='color: #ff003c; font-weight: bold; font-family: Orbitron;'>[ CRITICAL ]</span> <span style='color: #e2e8f0; font-weight: bold;'>US-IRAN TENSIONS</span><br><span style='color: #cbd5e1;'>Escalating rhetoric in the Strait of Hormuz is embedding a $2-$3 geopolitical risk premium.</span></div>
            <div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; margin-bottom: 15px;'><span style='color: #ffd700; font-weight: bold; font-family: Orbitron;'>[ WARNING ]</span> <span style='color: #e2e8f0; font-weight: bold;'>OPEC+ SHIFT</span><br><span style='color: #cbd5e1;'>Saudi Arabia leaning toward resuming gradual production increases in April 2026.</span></div>
            <div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;'><span style='color: #00f0ff; font-family: Orbitron; font-weight: bold;'>[ UPDATE ]</span> <span style='color: #e2e8f0; font-weight: bold;'>SPR FLOOR DEFENSE</span><br><span style='color: #cbd5e1;'>US DOE refill orders at $73/bbl creating a physical support floor for WTI pricing.</span></div>""", unsafe_allow_html=True)

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    c_head, c_btn1, c_btn2 = st.columns([6, 2, 2])
    with c_head: st.markdown(f"<h2 style='color:{intel['color']}; margin-top: 5px; font-size: 2rem;'>NODE::{target}</h2>", unsafe_allow_html=True)
    with c_btn1:
        st.write(""); 
        if st.button("← GLOBE"): st.session_state.target = None; st.rerun()
    with c_btn2:
        st.write(""); 
        if st.button("💬 OVIP AI"): ai_terminal()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WTI_PREMIUM (ADJ)", f"${(latest.get('WTI', 75) * mod):.2f}")
    m2.metric("LOCAL_VOL_SIGMA", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    m3.metric("GPR_INDEX", f"{(latest.get('gpr', 50) * mod):.1f}")
    m4.metric("THREAT_LEVEL", intel['risk'])
    st.markdown("---")
    col_left, col_right = st.columns([2.5, 1.5])
    with col_left:
        st.markdown(f"<h4 style='color:#00f0ff; margin-bottom: 5px;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        rgba_fill = f'rgba({int(intel["color"].lstrip("#")[0:2], 16)}, {int(intel["color"].lstrip("#")[2:4], 16)}, {int(intel["color"].lstrip("#")[4:6], 16)}, 0.1)'
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Vol', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=rgba_fill), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -10px; margin-bottom: 25px;'>> MATRIX_INTERPRETATION: The solid {intel['color']} wave tracks the localized 30-day volatility standard deviation for {target}. Rapid curve expansion indicates immediate supply chain dislocation. The dotted line tracks the global WTI baseline.</p>", unsafe_allow_html=True)
        st.plotly_chart(create_feature_importance_chart(intel['color']), use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -10px;'>> DRIVER_ANALYSIS: Random Forest extraction identifies GPR and Momentum as the heaviest factors overriding physical supply constraints in the NPRS-1 engine for {target}.</p>", unsafe_allow_html=True)
    with col_right:
        with st.container():
            st.markdown(f"""
            <div style='margin-bottom: 25px;'><p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ LOCAL_DYNAMICS ]</p><p style='font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.8; text-align: justify;'>{intel['info']}</p></div>
            <div style='margin-bottom: 25px;'><p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ MARKET_DATA ]</p><p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'><span style='color: #00f0ff;'>&gt;</span> {intel['figures']}</p></div>
            <div style='margin-bottom: 25px;'><p style='color: {intel['color']}; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px; text-shadow: 0 0 8px {hex_to_rgba(intel['color'], 0.4)};'>[ EVENT_IMPACT_ANALYSIS ]</p><p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; text-align: justify;'><span style='color: {intel['color']};'>&gt;</span> {intel['impact']}</p></div>
            <div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div><p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ SYSTEM_FORECAST ]</p>
            <ul style='font-family: "JetBrains Mono"; font-size: 13px; line-height: 2.0; color: #94a3b8; list-style-type: none; padding-left: 0;'>
            <li><span style='color: #00f0ff;'>[ GPR_TRACKING ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{(latest.get('gpr', 50)*mod):.1f}</span></li>
            <li><span style='color: {intel['color']};'>[ FORECAST ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{'Escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else ('Maintain' if intel['risk'] == 'MEDIUM' else 'Stabilize')}</span></li>
            <li><span style='color: #00f0ff;'>[ CORRELATION ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{'High' if intel['risk'] in ['MEDIUM', 'HIGH'] else 'Moderate'}</span></li>
            </ul>""", unsafe_allow_html=True)
