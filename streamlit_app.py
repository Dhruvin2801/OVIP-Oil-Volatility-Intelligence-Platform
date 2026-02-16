import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# Safely import AI engine
try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE SYSTEM CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="OVIP // Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Sleek, Modern Background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        overflow: hidden !important; /* NO SCROLL */
        height: 100vh !important;
    }
    
    /* Kill Top Padding for 1-Page Fit */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 98% !important;
    }
    header {visibility: hidden;}

    /* Animations: Fade In and Slide Up */
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .animated-panel {
        animation: slideUpFade 0.6s ease-out forwards;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.23);
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

# ==========================================
# 3. GLOBAL NODE DATABASE (30 Countries)
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'info': 'Major importer. High sensitivity to Brent/Dubai spreads. Expanding strategic reserves capacity.', 'color': '#10b981', 'mod': 0.95},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'info': 'Swing producer. Permian output highly elastic to WTI prices >$70/bbl.', 'color': '#38bdf8', 'mod': 1.0},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'info': 'Largest importer. Teapot refinery quotas dictate short-term physical demand cycles.', 'color': '#ef4444', 'mod': 1.1},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'info': 'Sanctioned exporter. Urals crude trading at structural discount. High shadow fleet reliance.', 'color': '#ef4444', 'mod': 1.25},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'info': 'De facto OPEC+ leader. ~3M bpd spare capacity acts as primary market shock absorber.', 'color': '#f59e0b', 'mod': 1.0},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'info': 'Key logistics hub. Fujairah port volumes serve as leading indicator for Middle East exports.', 'color': '#10b981', 'mod': 0.98},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'info': 'Sanctioned. Controls Strait of Hormuz chokepoint (21M bpd transit risk).', 'color': '#ef4444', 'mod': 1.3},
    'VENEZUELA': {'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'info': 'Heavy crude specialist. Production hampered by structural underinvestment and sanctions.', 'color': '#ef4444', 'mod': 1.15},
    'BRAZIL': {'lat': -14.23, 'lon': -51.92, 'risk': 'LOW', 'info': 'Non-OPEC growth driver. Pre-salt offshore fields lowering national breakeven costs.', 'color': '#10b981', 'mod': 0.9},
    'UK': {'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'info': 'Brent benchmark origin. Production naturally declining; regulatory uncertainty rising.', 'color': '#38bdf8', 'mod': 1.02},
    'NORWAY': {'lat': 60.47, 'lon': 8.46, 'risk': 'LOW', 'info': 'Highly reliable European supplier. Johan Sverdrup field provides heavy/medium baseload.', 'color': '#10b981', 'mod': 0.9},
    'NIGERIA': {'lat': 9.08, 'lon': 8.67, 'risk': 'HIGH', 'info': 'Light sweet crude exporter. Prone to onshore pipeline vandalism and force majeures.', 'color': '#ef4444', 'mod': 1.2},
    'ANGOLA': {'lat': -11.20, 'lon': 17.87, 'risk': 'MEDIUM', 'info': 'Exited OPEC. Deepwater production declining faster than replacement rate.', 'color': '#f59e0b', 'mod': 1.05},
    'LIBYA': {'lat': 26.33, 'lon': 17.22, 'risk': 'CRITICAL', 'info': 'Highly volatile production. Export terminals frequently blockaded by rival factions.', 'color': '#ef4444', 'mod': 1.4},
    'IRAQ': {'lat': 33.22, 'lon': 43.67, 'risk': 'HIGH', 'info': 'OPEC quota compliance issues. Federal vs. Kurdish export disputes restrict northern flows.', 'color': '#ef4444', 'mod': 1.15},
    'KUWAIT': {'lat': 29.31, 'lon': 47.48, 'risk': 'LOW', 'info': 'Stable OPEC producer. Al-Zour refinery shifting export mix from crude to products.', 'color': '#10b981', 'mod': 0.95},
    'QATAR': {'lat': 25.35, 'lon': 51.18, 'risk': 'LOW', 'info': 'LNG dominant, but condensate exports remain critical for Asian petrochemical margins.', 'color': '#10b981', 'mod': 0.95},
    'CANADA': {'lat': 56.13, 'lon': -106.34, 'risk': 'LOW', 'info': 'Heavy crude (WCS). Trans Mountain expansion finally relieving tidewater access bottlenecks.', 'color': '#10b981', 'mod': 0.98},
    'MEXICO': {'lat': 23.63, 'lon': -102.55, 'risk': 'MEDIUM', 'info': 'Maya crude exporter. State-run Pemex struggling with massive debt load and declining yields.', 'color': '#f59e0b', 'mod': 1.05},
    'GERMANY': {'lat': 51.16, 'lon': 10.45, 'risk': 'MEDIUM', 'event': 'Macro industrial slowdown reducing middle-distillate (diesel) demand.', 'color': '#38bdf8', 'mod': 1.0},
    'JAPAN': {'lat': 36.20, 'lon': 138.25, 'risk': 'MEDIUM', 'info': 'Major importer. Yen depreciation inflating local energy costs; nuclear restarts sluggish.', 'color': '#38bdf8', 'mod': 1.0},
    'SOUTH KOREA': {'lat': 35.90, 'lon': 127.76, 'risk': 'LOW', 'info': 'Top-tier refining hub. Highly dependent on Middle East sour crude imports.', 'color': '#10b981', 'mod': 0.95},
    'AUSTRALIA': {'lat': -25.27, 'lon': 133.77, 'risk': 'LOW', 'info': 'LNG powerhouse. Domestic refining capacity significantly reduced in recent years.', 'color': '#10b981', 'mod': 0.95},
    'ALGERIA': {'lat': 28.03, 'lon': 1.65, 'risk': 'MEDIUM', 'info': 'Sahara Blend exporter. Critical pipeline gas supplier to Southern Europe.', 'color': '#f59e0b', 'mod': 1.0},
    'EGYPT': {'lat': 26.82, 'lon': 30.80, 'risk': 'HIGH', 'info': 'Controls Suez Canal/SUMED pipeline. Houthi Red Sea attacks severely disrupting transit revenues.', 'color': '#ef4444', 'mod': 1.15},
    'TURKEY': {'lat': 38.96, 'lon': 35.24, 'risk': 'MEDIUM', 'info': 'Controls Bosphorus Strait. Key transit hub for Russian and Caspian crude.', 'color': '#f59e0b', 'mod': 1.05},
    'SOUTH AFRICA': {'lat': -30.55, 'lon': 22.93, 'risk': 'MEDIUM', 'info': 'Cape of Good Hope rerouting adding 10-14 days to East-West freight voyages.', 'color': '#f59e0b', 'mod': 1.1},
    'SINGAPORE': {'lat': 1.35, 'lon': 103.81, 'risk': 'LOW', 'info': 'Global pricing hub and chokepoint (Strait of Malacca). Bunkering demand remains robust.', 'color': '#10b981', 'mod': 0.95},
    'INDONESIA': {'lat': -0.78, 'lon': 113.92, 'risk': 'MEDIUM', 'info': 'Former OPEC member. Rising domestic demand absorbing local production.', 'color': '#f59e0b', 'mod': 1.0},
    'OMAN': {'lat': 21.51, 'lon': 55.92, 'risk': 'LOW', 'info': 'Key benchmark component (Dubai/Oman). Non-OPEC Middle East producer.', 'color': '#10b981', 'mod': 0.95}
}

# ==========================================
# 4. THE AI MODAL (POP-UP)
# ==========================================
@st.dialog("OVIP AI Terminal")
def ai_terminal():
    st.markdown("Ask the Daemon about macroeconomic trends or regional impacts.")
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "Online. How can I assist?"}]

    chat_box = st.container(height=300, border=False)
    with chat_box:
        for msg in st.session_state.chat_log:
            with st.chat_message(msg['role']): st.markdown(msg['content'])

    if prompt := st.chat_input("Enter query..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                if AI_AVAILABLE and st.session_state.vec is not None:
                    ctx = f"Target: {st.session_state.target if st.session_state.target else 'Global'}. {prompt}"
                    ans = get_ai_response(ctx, st.session_state.vec, st.session_state.tfidf, st.session_state.rag_df)
                else: ans = "AI Engine Offline."
                st.markdown(ans)
        st.session_state.chat_log.append({"role": "assistant", "content": ans})

# ==========================================
# 5. MAIN VIEW: VIBRANT GLOBE OR DASHBOARD
# ==========================================
if st.session_state.target is None:
    # --- VIBRANT GLOBE VIEW ---
    c_title, c_btn = st.columns([5, 1])
    c_title.markdown("<h2 style='margin-bottom:0;'>Global Threat Matrix</h2>", unsafe_allow_html=True)
    with c_btn:
        st.write("") # Spacing
        if st.button("💬 OVIP AI", use_container_width=True): ai_terminal()
    
    st.markdown("<p style='color:#94a3b8; font-size:14px;'>Click a strategic node to view localized intelligence.</p>", unsafe_allow_html=True)
    
    lats = [v['lat'] for v in COUNTRIES.values()]
    lons = [v['lon'] for v in COUNTRIES.values()]
    names = list(COUNTRIES.keys())
    colors = [v['color'] for v in COUNTRIES.values()]
    
    # Brighter, Vibrant Globe Design
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=14, color=colors, line=dict(width=1, color='#ffffff')),
        textfont = dict(family="Inter", size=12, color="#ffffff"),
        textposition = "top center"
    ))
    
    fig_globe.update_geos(
        projection_type="orthographic", 
        showcoastlines=True, coastlinecolor="#1e3a8a", # Deep blue oceans
        showland=True, landcolor="#0f172a",           # Dark slate land
        showocean=True, oceancolor="#020617",         # Very dark blue-black water
        showcountries=True, countrycolor="#334155", 
        bgcolor="rgba(0,0,0,0)",
        center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0) # Faces India
    )
    fig_globe.update_layout(height=650, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
        st.rerun()

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    # Animated Header
    st.markdown("<div class='animated-panel'>", unsafe_allow_html=True)
    col_title, col_btn_ai, col_btn_back = st.columns([4, 1, 1])
    col_title.markdown(f"<h2 style='color:{intel['color']}; margin:0;'>{target} Analytics</h2>", unsafe_allow_html=True)
    if col_btn_ai.button("💬 OVIP AI", use_container_width=True): ai_terminal()
    if col_btn_back.button("← Globe", use_container_width=True): st.session_state.target = None; st.rerun()
    
    st.markdown("---")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regional WTI Premium", f"${(latest.get('WTI', 75) * mod):.2f}")
    m2.metric("Local Volatility (Sigma)", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    m3.metric("Geopolitical Risk (GPR)", f"{(latest.get('gpr', 50) * mod):.1f}")
    m4.metric("Threat Level", intel['risk'])
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("") # Spacing

    # Content Row: Chart & Intelligence
    col_chart, col_intel = st.columns([2.5, 1.2])

    with col_chart:
        st.markdown("<div class='animated-panel' style='height: 480px;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#e2e8f0; margin-bottom: 10px;'>Volatility Impact Matrix</h4>", unsafe_allow_html=True)
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Volatility',
            line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=intel['color'] # Fills with the country color
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)',
            line=dict(color="#94a3b8", width=2, dash='dot')
        ), secondary_y=True)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
        fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI Price", color="#94a3b8", showgrid=False, secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_intel:
        st.markdown("<div class='animated-panel' style='height: 480px; overflow-y: auto;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#e2e8f0; border-bottom: 1px solid {intel['color']}; padding-bottom: 10px;'>Regional Intelligence</h4>", unsafe_allow_html=True)
        
        # Highly detailed, country-specific Intel
        st.markdown(f"""
        <div style='margin-top: 15px;'>
            <p style='color: {intel["color"]}; font-weight: 600; font-size: 1.1em;'>[ RISK ASSESSMENT: {intel["risk"]} ]</p>
            <p style='font-family: "Inter"; font-size: 15px; line-height: 1.6; color: #cbd5e1;'>
                {intel['info']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style='margin-top: 10px;'>
            <p style='color: #38bdf8; font-weight: 600; font-size: 1em;'>[ MACRO DRIVERS ]</p>
            <ul style='font-family: "Inter"; font-size: 14px; line-height: 1.6; color: #94a3b8;'>
                <li><b>GPR Tracking:</b> Localized index currently at {(latest.get('gpr', 50)*mod):.1f}.</li>
                <li><b>Volatility Forecast:</b> Expected to {'escalate rapidly' if intel['risk'] in ['HIGH', 'CRITICAL'] else ('maintain current range' if intel['risk'] == 'MEDIUM' else 'remain suppressed')} over the next 72 hours.</li>
                <li><b>WTI Correlation:</b> {'High' if intel['risk'] in ['MEDIUM', 'HIGH'] else 'Moderate'}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
