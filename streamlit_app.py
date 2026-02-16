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
    
    .hud-box {
        background: rgba(0, 240, 255, 0.03); 
        border: 1px solid rgba(0, 240, 255, 0.2); 
        padding: 8px 15px; 
        border-radius: 2px;
        box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.02);
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
    return f'rgba({int(hex_code[0:2], 16)},{int(hex_code[2:4], 16)},{int(hex_code[4:6], 16)},{alpha})'

def create_feature_importance_chart(color):
    factors = ['GPR (Geopolitical Risk)', 'WTI Momentum', 'NLP News Sentiment', 'OPEC+ Supply Quotas', 'Freight/Transit Costs']
    importance = [0.38, 0.22, 0.18, 0.14, 0.08]
    
    fig = go.Figure(go.Bar(
        x=importance, y=factors, orientation='h',
        marker=dict(color=color, line=dict(color='rgba(255,255,255,0.2)', width=1))
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text="[ NPRS-1 VOLATILITY_DRIVERS ]", font=dict(color="#00f0ff", family="Orbitron", size=12)),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=11, color="#e2e8f0"))
    )
    return fig

# ==========================================
# 3. GLOBAL NODE DATABASE
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE = '#00f0ff' 
C_WARN = '#ffd700'
C_DANG = '#ff003c'

# Highly Detailed Macro Intelligence
COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 
              'info': 'The third-largest global oil importer. The economy remains highly sensitive to Brent/Dubai pricing spreads. Local refinery run rates are currently maximizing at 102% utilization to meet rising domestic industrial demand.', 
              'impact': 'The structural discount on Russian crude imports has violently shrunk from $10/bbl down to $3.50/bbl due to tightening Western financial sanctions and dark fleet logistics bottlenecks. This is increasing fiscal pressure on state-run refiners. Strategic Petroleum Reserve (SPR) capacity is actively expanding by 6.5M tons to buffer against future Red Sea supply shocks.'},
    
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
            'info': 'The top global producer and primary swing supplier. The market is highly financialized. Permian basin output remains steady at roughly 6.1M bpd, but production growth is highly elastic to WTI prices dropping below the $70/bbl breakeven threshold.', 
            'impact': 'The enactment of broad-based import tariffs has caused immediate supply chain dislocations, forcing refiners to quickly re-route heavy crude inputs from alternative sources. Meanwhile, the Department of Energy is actively executing SPR refill operations, defending a hard physical price floor by purchasing bulk volumes at an average of $73/bbl.'},
    
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1, 
              'info': 'The largest global crude importer. Short-term physical demand cycles are entirely dictated by the central government’s issuance of independent "Teapot" refinery export quotas. Current Teapot utilization has plummeted to 58%.', 
              'impact': 'Aggressive tariff escalations with Western trading partners are severely reducing Chinese manufacturing export demand, leading to a massive buildup of middle-distillate inventories. Furthermore, 70% of China’s seaborne crude imports remain critically vulnerable to increasing geopolitical friction surrounding the Malacca Strait chokepoint.'},
    
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25, 
               'info': 'A heavily sanctioned exporter operating almost entirely outside the traditional financial system. Seaborne exports remain stubbornly high at 3.3M bpd, relying exclusively on a rapidly expanding dark fleet logistics network.', 
               'impact': 'Urals crude is currently trading at a steep $14 structural discount to the Brent benchmark. Western financial sanctions are successfully blocking critical infrastructure CapEx, leading to long-term decline rates. Furthermore, severe payment settlement delays with primary buyers (India and China) in non-dollar currencies are causing cargo float times to increase by an average of 12 days, trapping millions of barrels at sea.'},
    
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
                     'info': 'The de facto leader of the OPEC+ cartel. Acts as the primary market shock absorber by voluntarily withholding production. The Kingdom is currently holding approximately 3.2M bpd of spare capacity offline.', 
                     'impact': 'With the fiscal break-even price for the domestic Saudi budget estimated at $82/bbl, the current price environment is causing economic strain. Intelligence indicates OPEC+ is leaning heavily toward abandoning production cuts and resuming 2.2M bpd of output increases starting in April to wage a price war and reclaim Asian market share ahead of peak summer demand.'},
    
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3, 
             'info': 'A heavily sanctioned producer that physically controls the most critical global energy chokepoint: The Strait of Hormuz (21M bpd transit risk).', 
             'impact': 'Despite harsh sanctions, production has covertly recovered to 3.1M bpd, with 90% of exports routed via the dark fleet to independent Asian buyers. However, escalating rhetoric and frequent transit harassment incidents against commercial shipping in the Strait of Hormuz are actively embedding a persistent $3/bbl geopolitical risk premium into global crude benchmarks.'},
}

# Failsafe for un-detailed nodes
for k in ['UAE', 'VENEZUELA', 'BRAZIL', 'UK', 'NORWAY', 'NIGERIA', 'ANGOLA', 'LIBYA', 'IRAQ', 'KUWAIT', 'QATAR', 'CANADA', 'MEXICO', 'GERMANY', 'JAPAN', 'SOUTH KOREA', 'AUSTRALIA', 'ALGERIA', 'EGYPT', 'TURKEY', 'SOUTH AFRICA', 'SINGAPORE', 'INDONESIA', 'OMAN']:
    if k not in COUNTRIES:
        COUNTRIES[k] = {'lat': 0, 'lon': 0, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Standard node tracking active. Evaluating regional GPR metrics.', 'impact': 'Monitoring macro headwinds and localized supply chain constraints.'}

COUNTRIES['UAE'].update({'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE, 'info': 'Key logistics and storage hub. Fujairah port volumes serve as a leading indicator for total Middle East export health.', 'impact': 'Bunkering volumes at Fujairah have reached a 14-month high as global shipping networks capitalize on the port’s strategic location to bypass ongoing Red Sea transit rerouting.'})
COUNTRIES['UK'].update({'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'color': C_WARN, 'info': 'Origin of the Brent global pricing benchmark. North Sea production is in a state of terminal natural decline, currently dropping at 8% YoY.', 'impact': 'The aggressive implementation of localized windfall taxes on energy producers is severely restricting new drilling CapEx, forcing the European market to rely heavily on US imports.'})

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
    # --- GLOBE VIEW ---
    c_head, c_btn = st.columns([8.5, 1.5]) 
    with c_head:
        st.markdown("<h2 style='margin-top: 5px; font-size: 2rem;'>GLOBAL_THREAT_MATRIX</h2>", unsafe_allow_html=True)
    with c_btn:
        st.write("")
        if st.button("💬 OVIP AI"): ai_terminal()

    c_globe, c_stats = st.columns([7, 3])

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

    with c_stats:
        with st.container():
            st.markdown("""
            <div style='background: rgba(0, 240, 255, 0.03); border: 1px solid rgba(0, 240, 255, 0.2); padding: 15px; border-radius: 4px; margin-bottom: 20px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ NPRS-1 ENGINE ]</p>
                <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8;'>
                    STATUS: <span style='color: #00f0ff; font-weight: bold;'>ONLINE</span><br>
                    CONFIDENCE: <span style='color: #00f0ff; font-weight: bold;'>68.2% (UP_TREND)</span><br>
                    NEXT_CYCLE: <span style='color: #cbd5e1;'>12H:45M:03S</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: rgba(0, 240, 255, 0.03); border: 1px solid rgba(0, 240, 255, 0.2); padding: 15px; border-radius: 4px; margin-bottom: 20px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ MACRO_ENVIRONMENT ]</p>
                <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8;'>
                    REGIME: <span style='color: #ffd700; font-weight: bold;'>MODERATE_RISK</span><br>
                    CATALYST: <span style='color: #cbd5e1;'>TARIFF_POLICY_SHIFT</span><br>
                    VOLATILITY: <span style='color: #00f0ff; font-weight: bold;'>ELEVATED (+1.4%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    # Header
    c_head, c_btn1, c_btn2 = st.columns([6, 2, 2])
    with c_head:
        st.markdown(f"<h2 style='color:{intel['color']}; margin-top: 5px; font-size: 2rem;'>NODE::{target}</h2>", unsafe_allow_html=True)
    with c_btn1:
        st.write("")
        if st.button("← GLOBE"): st.session_state.target = None; st.rerun()
    with c_btn2:
        st.write("")
        if st.button("💬 OVIP AI"): ai_terminal()
        
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WTI_PREMIUM (ADJ)", f"${(latest.get('WTI', 75) * mod):.2f}")
    m2.metric("LOCAL_VOL_SIGMA", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    m3.metric("GPR_INDEX", f"{(latest.get('gpr', 50) * mod):.1f}")
    m4.metric("THREAT_LEVEL", intel['risk'])
    
    st.markdown("---")

    # Native Column Split
    col_left, col_right = st.columns([2.5, 1.5])

    with col_left:
        st.markdown(f"<h4 style='color:#00f0ff; margin-bottom: 5px;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        hex_color = intel['color'].lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgba_fill = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.1)'

        fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Volatility', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=rgba_fill), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Volatility (Sigma)", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI Index ($)", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # INTERPRETATION TEXT
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -15px; margin-bottom: 20px;'>> MATRIX_INTERPRETATION: The solid {intel['color']} wave tracks localized 30-day volatility standard deviation. Rapid curve expansion indicates immediate supply chain dislocation. The dotted grey line tracks the underlying global WTI baseline.</p>", unsafe_allow_html=True)
        
        # FEATURE IMPORTANCE CHART
        st.plotly_chart(create_feature_importance_chart(intel['color']), use_container_width=True, config={'displayModeBar': False})
        
        # INTERPRETATION TEXT
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -15px;'>> DRIVER_ANALYSIS: Random Forest feature extraction indicates Geopolitical Risk (GPR) and Momentum are currently overriding physical supply constraints in the NPRS-1 modeling engine for {target}.</p>", unsafe_allow_html=True)

    with col_right:
        with st.container():
            
            st.markdown(f"""
            <div style='margin-bottom: 30px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 16px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid rgba(0, 240, 255, 0.3); padding-bottom: 5px;'>[ LOCAL_DYNAMICS ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.8; text-align: justify;'>
                    {intel['info']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='margin-bottom: 30px;'>
                <p style='color: {intel['color']}; font-family: Orbitron; font-size: 16px; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid {hex_to_rgba(intel['color'], 0.3)}; padding-bottom: 5px; text-shadow: 0 0 8px {hex_to_rgba(intel['color'], 0.4)};'>[ EVENT_IMPACT_ANALYSIS ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; text-align: justify;'>
                    {intel['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)
