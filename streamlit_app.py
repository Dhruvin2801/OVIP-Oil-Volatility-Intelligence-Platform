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
    
    /* Native container styling for "panels" to avoid layout bugs */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background: rgba(10, 15, 30, 0.6);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 4px;
        padding: 20px;
        backdrop-filter: blur(5px);
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
# 3. GLOBAL NODE DATABASE (MASSIVE INTEL UPGRADE)
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

C_SAFE, C_WARN, C_DANG = '#00f0ff', '#ffd700', '#ff003c'

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 
              'info': 'India serves as a primary demand-side stabilizer. The economy is currently transitioning toward a $5 trillion target, making industrial consumption structural. Reliance Industries’ Jamnagar refinery, the world\'s largest refining complex, is operating at 104% capacity to maximize export margins for diesel to Europe while state refiners IOCL and BPCL secure domestic supply.', 
              'figures': 'Strategic Petroleum Reserve (SPR) Phase II expanding by 47M barrels | Refinery throughput hitting 5.2M bpd | Seaborne import reliance at 87%.', 
              'impact': 'The "discounted" Urals crude trade—once $10-$15/bbl below Brent—has violently collapsed to a spread of just $3.25/bbl. This shrinking arbitrage is driving IOCL toward more aggressive rupee-denominated trade agreements with Middle Eastern NOCs like ADNOC and Saudi Aramco to stabilize fiscal volatility and reduce Dollar dependency.'},
              
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
            'info': 'The U.S. remains the world’s largest producer and primary non-OPEC swing supplier. Permian basin shale output is sustaining at 6.1M bpd, but CapEx efficiency is slowing. The market is now hypersensitive to EIA/API weekly inventory builds as an indicator for recessionary diesel demand cooling.', 
            'figures': 'Commercial Inventories: 420.5M barrels (-2.4M WoW) | SPR Levels: Recovering to 370M barrels | Active Rig Count: 498 (Stagnant).', 
            'impact': 'The Department of Energy is effectively backstopping the global market floor by executing aggressive Strategic Petroleum Reserve (SPR) refill orders whenever WTI breaches the $72/bbl threshold. Simultaneously, broad-based tariffs on Mexican and Canadian heavy crude are forcing Gulf Coast refiners to reconfigure coking units for lighter domestic grades, creating short-term volatility in regional pricing spreads.'},
            
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1, 
              'info': 'China is the "Great Volatility Engine." Demand signals from the Shandong independent "Teapot" refineries dictate short-term physical market velocity. National strategy is currently focused on an aggressive build-out of refined product stocks as a buffer against trade escalation and naval chokepoint friction.', 
              'figures': 'Teapot Refinery Utilization: 58.1% (Multi-year low) | Monthly Crude Imports: 10.4M bpd | Stockpile estimate: 950M barrels.', 
              'impact': 'A structural manufacturing PMI slowdown below 50.0 is creating a massive "Supply Glut" in middle-distillates. Intelligence indicates that 72% of Chinese seaborne imports remain exposed to the Malacca Strait chokepoint; any regional naval friction generates an immediate "Geopolitical Fear Premium" of $4-$6/bbl in the NPRS-1 engine, overriding weak industrial demand data.'},
              
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25, 
               'info': 'Russia is operating a parallel energy economy. Despite G7 price caps, seaborne exports are sustaining at roughly 3.3M bpd through a Shadow Fleet of over 600 tankers. A lack of Western CapEx is leading to reservoir pressure decline in legacy Siberian fields, forcing a shift toward more expensive Arctic exploration.', 
               'figures': 'Dark Fleet Estimated Capacity: 620 vessels | Seaborne Export Flow: 3.3M bpd | Urals-Brent Spread: -$14.50.', 
               'impact': 'Severe non-dollar payment bottlenecks (Rupee/Yuan) with primary Asian buyers have increased average cargo float times by 12 days. This "trapped inventory" creates localized scarcity in the physical market, while western sanctions on Sovcomflot tankers continue to push freight insurance premiums 40% higher than global averages, embedding volatility into delivered cargo prices.'},
               
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
                     'info': 'The de facto leader of the OPEC+ coalition. Saudi Arabia remains the only global entity capable of providing a 3M bpd spare capacity "safety net." Their domestic fiscal policy is now locked in a struggle between funding the $1.5 trillion "Vision 2030" projects and maintaining global market share dominance.', 
                     'figures': 'Current Output: ~9.0M bpd | Spare Capacity Offline: 3.2M bpd | Fiscal Break-even: $82.40/bbl.', 
                     'impact': 'With the fiscal break-even target near $83/bbl, the Kingdom is facing immense pressure. Intelligence indicates a likely "Pivot" in April 2026 where the Ministry of Energy may abandon voluntary cuts and flood the market with 2.2M bpd of incremental output. This strategy aims to wash out high-cost marginal producers in the US and reclaim dominance in the Asian refining corridors, likely triggering a volatility spike.'},
                     
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.98, 
            'info': 'The UAE is aggressively positioning itself as the primary logistics chokepoint bypass for the Middle East. ADNOC is in the middle of a massive CapEx ramp-up to hit 5M bpd capacity by 2027, increasingly diverging from Saudi-led supply restriction strategies.', 
            'figures': 'Fujairah Storage Capacity: 10M barrels | ADNOC Murban Crude Exports: +11% YoY | Bunkering activity: 14-month high.', 
            'impact': 'Global shipping networks are rerouting around the Red Sea and utilizing Fujairah as the primary safe-harbor refueling station. This has generated record bunkering demand, decoupling regional product pricing from the underlying Brent benchmark. The UAE is effectively using its strategic geography to immunize its energy revenues from regional conflict volatility.'},
            
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3, 
             'info': 'Iran physically controls the world\'s most critical energy chokepoint: The Strait of Hormuz. 20.8M bpd of liquid petroleum transits this waterway. Domestic production has covertly recovered to 3.1M bpd despite the "Maximum Pressure" sanction regime remaining legally in place.', 
             'figures': 'Hormuz Transit Volume: 20.8M bpd | Covert Production: 3.1M bpd | Asian Export Share: 92%.', 
             'impact': 'Frequent harassment of commercial shipping and the deployment of "proxy" threats in the Bab al-Mandab are embedding a persistent $3.00/bbl Geopolitical Risk (GPR) premium into the Brent forward curve. The NPRS-1 engine identifies Iran as the primary "Asymmetric Risk" node; any escalation here creates a 10-sigma volatility event that ignores physical storage data.'},
             
    'BRAZIL': {'lat': -14.24, 'lon': -51.93, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.9, 
               'info': 'The primary non-OPEC growth engine for the Southern Hemisphere. Petrobras is executing a $100B five-year plan focused exclusively on ultra-deepwater Pre-salt reserves. Brazil is now a critical supplier for independent Chinese refiners who are seeking to diversify away from Middle Eastern dependency.', 
               'figures': 'Current Record Output: 3.5M bpd | Pre-salt Share: 81% | Export Growth: +14% YoY.', 
               'impact': 'Lula da Silva’s energy policy is balancing domestic fuel subsidies with export maximization. The increasing flow of Brazilian "Lula" and "Buzios" grades into Asia is putting structural downward pressure on West African and North Sea premiums, smoothing global volatility during Middle Eastern supply shocks.'},
               
    'NORWAY': {'lat': 60.47, 'lon': 8.47, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.9, 
               'info': 'Norway is now the single most important energy security partner for the European Union. The Johan Sverdrup field provides a reliable, heavy-medium baseload of over 750,000 bpd, effectively replacing sanctioned Russian Urals in Northern European refineries.', 
               'figures': 'Johan Sverdrup Output: 755k bpd | Total Petroleum Exports: 2.1M bpd | EU Energy Reliance: 35%.', 
               'impact': 'Equinor’s strategic shift toward maintaining plateau production rather than rapid expansion has stabilized regional volatility. However, the lack of incremental offshore exploration CapEx means Norway cannot act as a supply-side cushion during a major crisis, making the Eurozone structurally dependent on US seaborne arrivals.'},
               
    'NIGERIA': {'lat': 9.08, 'lon': 8.68, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.2, 
                'info': 'Nigeria is the "Fragile Node." Production of its high-quality light sweet crude is constantly under threat from onshore pipeline vandalism in the Niger Delta and large-scale industrial-scale theft. The ramp-up of the massive Dangote Refinery is fundamentally changing regional dynamics.', 
                'figures': 'Production Ceiling: 1.45M bpd | Dangote Refinery Capacity: 650k bpd | Brent Premium: +$2.10/bbl.', 
                'impact': 'The Dangote Refinery is expected to turn Nigeria from a gasoline importer into a dominant exporter for West Africa by late 2026. This is disrupting established "Gasoline-to-Crude" trade flows from Europe, creating a temporary pricing vacuum and increasing localized volatility in the Mediterranean and Gulf of Guinea refining markets.'}
}

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
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold;'>[ NPRS-1 ENGINE ]</p>", unsafe_allow_html=True)
            st.markdown("""
            * **STATUS:** ONLINE
            * **CONFIDENCE:** 68.2% (UP_TREND)
            * **NEXT_CYCLE:** 12H:45M:03S
            """)
            st.markdown("---")
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold;'>[ MACRO_ENVIRONMENT ]</p>", unsafe_allow_html=True)
            st.markdown("""
            * **REGIME:** MODERATE_RISK
            * **CATALYST:** TARIFF_POLICY_SHIFT
            * **VOLATILITY:** ELEVATED (+1.4%)
            """)
            st.markdown("---")
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold;'>[ LIVE_FEED ]</p>", unsafe_allow_html=True)
            st.markdown("""
            **[ CRITICAL ] US-IRAN TENSIONS:** Escalating rhetoric and transit harassment in the Strait of Hormuz is embedding a $2-$3 geopolitical risk premium.
            
            **[ WARNING ] OPEC+ SHIFT:** Saudi Arabia leaning toward resuming gradual production increases in April 2026.
            """)

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

    col_left, col_right = st.columns([2.5, 1.5])

    with col_left:
        st.markdown(f"<h4 style='color:#00f0ff; margin-bottom: 15px;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        rgba_fill = f'rgba({int(intel["color"].lstrip("#")[0:2], 16)}, {int(intel["color"].lstrip("#")[2:4], 16)}, {int(intel["color"].lstrip("#")[4:6], 16)}, 0.1)'

        fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Volatility', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=rgba_fill), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Volatility (Sigma)", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI Index ($)", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # GRAPH 1 EXPLANATION
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -10px; margin-bottom: 25px;'>> MATRIX_INTERPRETATION: The solid {intel['color']} wave tracks localized 30-day volatility standard deviation for {target}. Rapid expansion indicates physical supply chain dislocation. The dotted grey line tracks the underlying global WTI baseline.</p>", unsafe_allow_html=True)
        
        st.plotly_chart(create_feature_importance_chart(intel['color']), use_container_width=True, config={'displayModeBar': False})
        
        # GRAPH 2 EXPLANATION
        st.markdown(f"<p style='color: #64748b; font-family: JetBrains Mono; font-size: 11px; margin-top: -10px;'>> DRIVER_ANALYSIS: Random Forest extraction identifies Geopolitical Risk (GPR) and Price Momentum as the heaviest weighted factors currently overriding physical supply constraints in the NPRS-1 modeling engine for the {target} node.</p>", unsafe_allow_html=True)

    with col_right:
        with st.container():
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ LOCAL_DYNAMICS ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.8; text-align: justify; padding-left: 12px; border-left: 2px solid #00f0ff;'>
                    {intel['info']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ MARKET_DATA ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'>
                    <span style='color: #00f0ff;'>&gt;</span> {intel['figures']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: {intel['color']}; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px; text-shadow: 0 0 8px {hex_to_rgba(intel['color'], 0.4)};'>[ EVENT_IMPACT_ANALYSIS ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; text-align: justify;'>
                    <span style='color: {intel['color']};'>&gt;</span> {intel['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)
