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
        padding-top: 1.5rem !important; 
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        margin-top: -30px !important; 
    }
    header {visibility: hidden;}

    .stApp::before {
        content: " "; display: block; position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 20, 0, 0.1) 50%);
        z-index: 999; background-size: 100% 4px; pointer-events: none;
    }

    .animated-panel {
        background: rgba(10, 15, 30, 0.6); 
        border: 1px solid rgba(0, 240, 255, 0.2); 
        border-radius: 4px; 
        padding: 15px;
        backdrop-filter: blur(5px);
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
        background: rgba(0, 240, 255, 0.05) !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
        border-radius: 2px !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        transition: all 0.2s ease;
        padding: 5px 15px !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background: #00f0ff !important;
        color: #000 !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }
    
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.3); border-radius: 2px; }
    
    /* Engine Stat Boxes for Home Page */
    .stat-box {
        background: rgba(0, 240, 255, 0.03); 
        border: 1px solid rgba(0, 240, 255, 0.2); 
        padding: 12px; 
        border-radius: 2px;
        box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.02);
        height: 100%;
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

def create_country_sentiment_chart(data, color):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Score'], marker_color=color, marker_line_width=0))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', tickfont=dict(color="#94a3b8")), 
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)', tickfont=dict(color="#94a3b8")),
    )
    return fig

# ==========================================
# 3. GLOBAL NODE DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE = '#00f0ff' 
C_WARN = '#ffd700'
C_DANG = '#ff003c'

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95,
              'info': 'Third largest global oil importer. High sensitivity to Brent/Dubai spreads.',
              'figures': 'Refinery run rates at 102%. Strategic Petroleum Reserve (SPR) capacity expanding by 6.5M tons.',
              'impact': 'Russian crude import discounts have shrunk from $10/bbl to $3.50/bbl, increasing fiscal pressure.'},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0,
            'info': 'Top global producer & swing supplier. Highly financialized market.',
            'figures': 'Permian output steady at 6.1M bpd. Commercial crude inventories down 2.4M barrels.',
            'impact': 'Broad-based import tariffs enacted. Refiners re-routing heavy crude inputs. SPR refill operations purchasing at $73/bbl average.'},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1,
              'info': 'Largest global crude importer. Teapot quotas dictate short-term cycles.',
              'figures': 'Independent refinery utilization dropped to 58%. LNG imports up 14% YoY.',
              'impact': 'Tariff escalation reducing manufacturing export demand. 70% of crude imports remain vulnerable to Malacca Strait chokepoint risks.'},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25,
               'info': 'Heavily sanctioned exporter utilizing vast dark fleet logistics.',
               'figures': 'Urals crude trading at $14 discount to Brent. Seaborne exports at 3.3M bpd.',
               'impact': 'Western financial sanctions blocking infrastructure CapEx. Payment settlement delays with India/China causing cargo float times to increase by 12 days.'},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0,
                     'info': 'De facto OPEC+ leader. Acts as primary market shock absorber.',
                     'figures': 'Holding ~3.2M bpd of spare capacity. Fiscal break-even estimated at $82/bbl.',
                     'impact': 'OPEC+ leaning toward resuming 2.2M bpd output increases starting in April to reclaim Asian market share ahead of summer demand.'},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3,
             'info': 'Sanctioned producer controlling the critical Strait of Hormuz.',
             'figures': 'Production recovering to 3.1M bpd. 90% of exports routed via dark fleet to independent Asian buyers.',
             'impact': 'Strait of Hormuz transit harassment incidents are embedding a persistent $3/bbl geopolitical risk premium into global benchmarks.'}
}

for k in ['UAE', 'VENEZUELA', 'BRAZIL', 'UK', 'NORWAY', 'NIGERIA', 'ANGOLA', 'LIBYA', 'IRAQ', 'KUWAIT', 'QATAR', 'CANADA', 'MEXICO', 'GERMANY', 'JAPAN', 'SOUTH KOREA', 'AUSTRALIA', 'ALGERIA', 'EGYPT', 'TURKEY', 'SOUTH AFRICA', 'SINGAPORE', 'INDONESIA', 'OMAN']:
    if k not in COUNTRIES:
        COUNTRIES[k] = {'lat': 0, 'lon': 0, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Standard node tracking active.', 'figures': 'Data aggregating...', 'impact': 'Monitoring macro headwinds.'}

COUNTRIES['UAE'].update({'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE, 'figures': 'Fujairah bunkering at 14-month high.', 'impact': 'Capitalizing on Red Sea rerouting.'})
COUNTRIES['VENEZUELA'].update({'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'color': C_DANG, 'figures': 'Output stalled at 800k bpd.', 'impact': 'Sanctions snapback limiting diluent imports.'})
COUNTRIES['UK'].update({'lat': 55.37, 'lon': -3.43, 'color': C_WARN, 'figures': 'North Sea decline rate at 8% YoY.', 'impact': 'Windfall tax restricting new drilling.'})

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
    # --- HOME VIEW: GLOBE & LIVE FEED & STATS ---
    
    c_title, c_btn = st.columns([8, 2])
    with c_title:
        st.markdown("<h2 style='margin-top: 10px; font-size: 2rem;'>GLOBAL_THREAT_MATRIX</h2>", unsafe_allow_html=True)
    with c_btn:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI UPLINK"): ai_terminal()

    # Layout: Globe on Left (75%), News/Stats on Right (25%)
    c_globe, c_right = st.columns([7.5, 2.5]) 
    
    with c_globe:
        lats = [v['lat'] for v in COUNTRIES.values()]
        lons = [v['lon'] for v in COUNTRIES.values()]
        names = list(COUNTRIES.keys())
        colors = [v['color'] for v in COUNTRIES.values()]
        
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
        
        fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
        if event and "selection" in event and event["selection"]["points"]:
            st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
            st.rerun()

    with c_right:
        # Top right: The requested NPRS/Macro blocks
        st.markdown("""
        <div class='stat-box' style='margin-bottom: 15px;'>
            <div style='color: #00f0ff; font-family: Orbitron; font-size: 11px; margin-bottom: 8px; letter-spacing: 1px;'>[ NPRS-1 ENGINE ]</div>
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'>
                STATUS: <span style='color: #00f0ff;'>ONLINE</span><br>
                CONFIDENCE: <span style='color: #00f0ff;'>68.2% (UP_TREND)</span><br>
                NEXT_CYCLE: 12H:45M:03S
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='stat-box' style='margin-bottom: 20px;'>
            <div style='color: #00f0ff; font-family: Orbitron; font-size: 11px; margin-bottom: 8px; letter-spacing: 1px;'>[ MACRO_ENVIRONMENT ]</div>
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'>
                REGIME: <span style='color: #ffd700;'>MODERATE_RISK</span><br>
                CATALYST: TARIFF_POLICY_SHIFT<br>
                VOLATILITY: <span style='color: #00f0ff;'>ELEVATED (+1.4%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bottom right: Scrollable News Feed
        st.markdown("<div class='animated-panel' style='height: 420px; overflow-y: auto; padding: 15px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00f0ff; border-bottom: 1px solid rgba(0,240,255,0.3); padding-bottom: 5px; font-size: 1rem;'>LIVE_MACRO_FEED</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6;'>
            <div style='margin-top: 10px; margin-bottom: 20px;'>
                <span style='color: #ff003c; font-weight: bold; font-family: Orbitron;'>[ CRITICAL ] US-IRAN TENSIONS</span><br>
                <span style='color: #cbd5e1;'>Escalating rhetoric and transit harassment in the Strait of Hormuz is actively embedding a $2-$3 geopolitical risk premium into Brent crude pricing.</span>
            </div>
            <div style='margin-bottom: 20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;'>
                <span style='color: #ffd700; font-weight: bold; font-family: Orbitron;'>[ WARNING ] OPEC+ POLICY SHIFT</span><br>
                <span style='color: #cbd5e1;'>Saudi Arabia and the UAE are leaning toward resuming gradual production increases starting in April 2026 to reclaim market share ahead of summer demand.</span>
            </div>
            <div style='border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;'>
                <span style='color: #00f0ff; font-weight: bold; font-family: Orbitron;'>[ UPDATE ] IEA OVERSUPPLY FORECAST</span><br>
                <span style='color: #cbd5e1;'>Global oil inventories swelled by 477M barrels last year. A severe supply glut is projected through late 2026, led by non-OPEC output.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES.get(target, COUNTRIES['USA'])
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    # Header Row
    c_head, c_btn1, c_btn2 = st.columns([6, 2, 2])
    with c_head:
        st.markdown(f"<h2 style='color:{intel['color']}; margin-top: 10px; font-size: 2rem;'>NODE::{target}</h2>", unsafe_allow_html=True)
    with c_btn1: 
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("← RETURN TO GLOBE"): st.session_state.target = None; st.rerun()
    with c_btn2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("💬 OVIP AI UPLINK"): ai_terminal()
        
    # Metrics Row
    st.markdown("<div class='animated-panel' style='margin-bottom: 15px; margin-top: 10px; padding: 10px 20px;'>", unsafe_allow_html=True)
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

    # Content Row 1: Matrix & Real Event Impact
    c_chart1, c_text1 = st.columns([2.5, 1.5])

    with c_chart1:
        st.markdown("<div class='animated-panel' style='height: 380px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#00f0ff;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Vol', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=hex_to_rgba(intel['color'], 0.1)), secondary_y=False)
        fig1.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)

        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig1.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig1.update_yaxes(title_text="Volatility (Sigma)", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig1.update_yaxes(title_text="WTI Index", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_text1:
        # COMPLETELY REBUILT PANEL: Pure Streamlit Markdown, no complex HTML divs wrapping the variables
        st.markdown("<div class='animated-panel' style='height: 380px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{intel['color']}; border-bottom: 1px solid {hex_to_rgba(intel['color'], 0.3)}; padding-bottom: 10px;'>LOCAL_EVENT_IMPACT</h4>", unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: #00f0ff; font-family: JetBrains Mono; font-size: 13px; font-weight: bold; margin-top: 10px;'>[ MARKET_DATA ]</p>", unsafe_allow_html=True)
        st.write(intel['figures'])
        
        st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 15px 0;'>", unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: {intel['color']}; font-family: JetBrains Mono; font-size: 13px; font-weight: bold;'>[ EVENT_IMPACT ]</p>", unsafe_allow_html=True)
        st.write(intel['impact'])
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Content Row 2: Sentiment Graph & Deep Intel
    c_chart2, c_text2 = st.columns([2.5, 1.5])
    
    with c_chart2:
        st.markdown("<div class='animated-panel' style='height: 280px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#00f0ff;'>REGIONAL_SENTIMENT_INDEX</h4>", unsafe_allow_html=True)
        loc_df = df_main.tail(60).copy()
        loc_df['Score'] = loc_df['Score'] * (2 - mod)
        st.plotly_chart(create_country_sentiment_chart(loc_df, intel['color']), use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_text2:
        st.markdown("<div class='animated-panel' style='height: 280px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#00f0ff; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 10px;'>DEEP_INTEL</h4>", unsafe_allow_html=True)
        
        st.write(f"> {intel['info']}")
        st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 10px 0;'>", unsafe_allow_html=True)
        st.write(f"**GPR_TRACKING:** {(latest.get('gpr', 50)*mod):.1f}")
        st.write(f"**72H_FORECAST:** {'Escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else ('Maintain' if intel['risk'] == 'MEDIUM' else 'Stabilize')}")
        
        st.markdown("</div>", unsafe_allow_html=True)
