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

# CSS for styling colors, fonts, and containers, NOT layout structure.
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
    # Simulated Feature Importance Data
    factors = ['Geo-Risk (GPR)', 'WTI Momentum', 'NLP Sentiment', 'OPEC+ Supply', 'Freight Cost']
    importance = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    fig = go.Figure(go.Bar(
        x=importance, y=factors, orientation='h',
        marker=dict(color=color, line=dict(color='rgba(255,255,255,0.2)', width=1))
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text="VOLATILITY_DRIVERS", font=dict(color="#00f0ff", family="Orbitron", size=14)),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(family="JetBrains Mono", size=12, color="#e2e8f0"))
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
              'info': 'As the world\'s third-largest oil importer, India\'s industrial base is structurally dependent on stable Brent pricing. Reliance Industries Jamnagar complex, the world\'s largest refining hub, is currently operating at 104% utilization to capture widening middle-distillate cracks. The Indian government is aggressively prioritizing Rupee-denominated trade to mitigate Dollar liquidity shocks.', 
              'figures': 'Strategic Reserve Capacity: +6.5M tons expansion in progress | Daily Consumption: ~5.4M barrels | Refinery Run Rate: 102% average across state-run units.', 
              'impact': 'The historic $10-$15/bbl discount on Russian Urals crude has violently compressed to just $3.50/bbl due to secondary sanction enforcement on the "Dark Fleet." This shrinkage is forcing Indian state refiners like IOCL to pivot back toward Saudi term-contracts, creating a localized supply-chain bottleneck and increasing fiscal volatility for New Delhi.'},
              
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
            'info': 'The United States maintains its position as the top global producer and the primary non-OPEC swing supplier. Permian Basin output remains robust at 6.1M bpd, but CapEx efficiency is plateauing as Tier-1 drilling locations are exhausted. The market remains hyper-fixated on the SPR refill strategy as a hard price floor.', 
            'figures': 'Commercial Inventories: 420M barrels | Current SPR Level: 370M barrels | Active Rig Count: 498 (Stagnant).', 
            'impact': 'The Department of Energy is effectively backstopping the global market by executing refill orders at an average price of $73/bbl. Simultaneously, the threat of broad tariffs on Mexican heavy crude is causing massive dislocations for Gulf Coast refiners, who must now compete for seaborne heavy grades from South America, injecting a $2/bbl regional premium into WTI-Midland spreads.'},
            
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'color': C_DANG, 'mod': 1.1, 
              'info': 'China is the primary driver of global crude demand volatility. Physical market velocity is dictated by the government’s quota system for independent "Teapot" refineries in Shandong. Macro industrial data indicates a buildup of diesel inventories as domestic manufacturing export demand cools under international trade friction.', 
              'figures': 'Teapot Refinery Utilization: 58.1% (Multi-year low) | Monthly Crude Imports: 10.4M bpd | Strategic Stockpile Estimate: 950M barrels.', 
              'impact': 'Escalating trade tariffs with Western partners are creating a structural demand gap, with refining throughput dropping significantly below 2024 levels. Critically, 72% of Chinese seaborne imports are exposed to the Malacca Strait chokepoint; any regional naval friction generates an immediate "Geopolitical Fear Premium" in the NPRS-1 engine that overrides physical demand destruction data.'},
              
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.25, 
               'info': 'Russia continues to operate a shadow energy economy. Despite G7 price caps, seaborne exports are sustaining at roughly 3.3M bpd through an expansive Dark Fleet network of over 600 tankers. A systemic lack of Western technology is leading to reservoir pressure decline in mature Siberian fields.', 
               'figures': 'Urals-Brent Spread: -$14.50 | Seaborne Export Volume: 3.3M bpd | Dark Fleet Capacity: ~650 vessels.', 
               'impact': 'Financial sanctions have caused severe non-dollar payment bottlenecks with Asian buyers. Cargo "float times" have increased by 12 days on average, effectively trapping 40M barrels of oil at sea. This artificial scarcity in the physical prompt market is driving localized volatility, while western sanctions on Sovcomflot tankers push freight insurance premiums 40% higher than global averages.'},
               
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 
                     'info': 'The de facto leader of the OPEC+ coalition. Saudi Arabia is the only global entity capable of maintaining a 3.2M bpd spare capacity buffer. National strategy is currently balancing the $82/bbl fiscal break-even required for Vision 2030 projects against the risk of permanent market share loss to US shale.', 
                     'figures': 'Spare Capacity: 3.2M bpd | Output: ~9.0M bpd | National Fiscal Break-even: $82.40/bbl.', 
                     'impact': 'With global benchmarks trading below the fiscal break-even, intelligence suggests a likely "Policy Pivot" in Q2 2026. The Kingdom is expected to abandon voluntary cuts and flood the market with incremental output to wash out marginal producers. This "Price War" scenario would trigger a massive volatility event, likely driving a 10-sigma move in the NPRS-1 engine probabilities.'},
                     
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.98, 
            'info': 'The UAE is positioning Fujairah as the primary global logistics and storage chokepoint bypass. ADNOC is executing a massive CapEx expansion to hit 5M bpd capacity by 2027, increasingly diverging from traditional Saudi-led restriction strategies. Bunkering activity has hit a 14-month high.', 
            'figures': 'Fujairah Storage: 10M barrels | ADNOC Murban Exports: +11% YoY | Capacity Target: 5M bpd.', 
            'impact': 'Global shipping networks are rerouting around the Red Sea and utilizing Fujairah as a safe-harbor refueling station. This surge in demand has decoupled regional product pricing from global benchmarks. The UAE is successfully using its strategic geography to immunize its energy revenues from the broader volatility caused by the Strait of Hormuz conflict zone.'},
            
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'color': C_DANG, 'mod': 1.3, 
             'info': 'Iran controls the Strait of Hormuz, through which 20.8M bpd—over 20% of global consumption—must pass. Production has covertly surged to 3.1M bpd despite the "Maximum Pressure" sanction regime. Regional proxy tensions remain at an all-time high.', 
             'figures': 'Hormuz Transit Volume: 20.8M bpd | Covert Export Share: 92% | Production: 3.1M bpd.', 
             'impact': 'Persistent harassment of commercial shipping in the Strait is embedding a permanent $3.00/bbl Geopolitical Risk premium into global pricing. The NPRS-1 engine identifies Iran as the primary "Asymmetric Risk" node; any kinetic escalation here would instantly generate a 20% volatility spike that ignores all physical demand and storage data.'},
             
    'UK': {'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.02, 
           'info': 'Origin of the Brent global pricing benchmark. North Sea production is in terminal natural decline, dropping at 8% YoY. Local energy policy is currently prioritizing windfall taxation over offshore exploration CapEx.', 
           'figures': 'North Sea Decline Rate: 8% YoY | Active Offshore Rigs: Historic Low | Brent Share of Global Trade: 60%.', 
           'impact': 'The aggressive windfall tax regime is irreversibly eroding domestic production capabilities. This is forcing the UK and broader European market to rely heavily on US shale imports, making the region structurally vulnerable to transatlantic freight volatility and Port of Houston logistical bottlenecks.'},
           
    'OMAN': {'lat': 21.51, 'lon': 55.92, 'risk': 'LOW', 'color': C_SAFE, 'mod': 0.95, 
             'info': 'Strategic non-OPEC node. Origin of the Dubai/Oman benchmark for Asian trade. Oman has successfully expanded its crude blending operations and strategic storage capacity at Duqm.', 
             'figures': 'Daily Output: ~1.05M bpd | Duqm Storage Capacity: 25M barrels | Non-OPEC Share of Asia trade: High.', 
             'impact': 'The completion of the Duqm bypass pipelines allows Omani crude to be exported without entering the Strait of Hormuz. This "Security Premium" is attracting Asian buyers like Sinopec and PetroChina, who are paying a localized premium for Omani grades to avoid the insurance spikes associated with Hormuz transit.'}
}

for k in ['BRAZIL', 'NORWAY', 'NIGERIA', 'ANGOLA', 'LIBYA', 'IRAQ', 'KUWAIT', 'QATAR', 'CANADA', 'MEXICO', 'GERMANY', 'JAPAN', 'SOUTH KOREA', 'AUSTRALIA', 'ALGERIA', 'EGYPT', 'TURKEY', 'SOUTH AFRICA', 'SINGAPORE', 'INDONESIA']:
    if k not in COUNTRIES:
        COUNTRIES[k] = {'lat': 0, 'lon': 0, 'risk': 'MEDIUM', 'color': C_WARN, 'mod': 1.0, 'info': 'Standard node tracking active. The AI engine is evaluating regional Geopolitical Risk (GPR) metrics and baseline production capacity.', 'figures': 'Data aggregating from primary satellite and manifest sources...', 'impact': 'Currently monitoring overarching global macro headwinds, structural demand destruction vectors, and localized supply chain constraints.'}

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
        # PURE STREAMLIT CONTAINER WITH COLORIZED MARKDOWN
        with st.container():
            # NPRS-1 ENGINE BLOCK
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ NPRS-1 ENGINE ]</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; margin-bottom: 25px;'>
                STATUS: <span style='color: #00f0ff; font-weight: bold;'>ONLINE</span><br>
                CONFIDENCE: <span style='color: #00f0ff; font-weight: bold;'>68.2% (UP_TREND)</span><br>
                NEXT_CYCLE: <span style='color: #cbd5e1;'>12H:45M:03S</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            
            # MACRO ENVIRONMENT BLOCK
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>[ MACRO_ENVIRONMENT ]</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.8; margin-bottom: 25px;'>
                REGIME: <span style='color: #ffd700; font-weight: bold;'>MODERATE_RISK</span><br>
                CATALYST: <span style='color: #cbd5e1;'>TARIFF_POLICY_SHIFT</span><br>
                VOLATILITY: <span style='color: #00f0ff; font-weight: bold;'>ELEVATED (+1.4%)</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div>", unsafe_allow_html=True)

            # LIVE FEED BLOCK
            st.markdown("<p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 15px;'>[ LIVE_FEED ]</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6; margin-bottom: 15px;'>
                <span style='color: #ff003c; font-weight: bold; font-family: Orbitron;'>[ CRITICAL ]</span> <span style='color: #e2e8f0; font-weight: bold;'>US-IRAN TENSIONS</span><br>
                <span style='color: #cbd5e1;'>Escalating rhetoric and transit harassment in the Strait of Hormuz is embedding a $2-$3 geopolitical risk premium.</span>
            </div>
            <div style='font-family: JetBrains Mono; font-size: 12px; line-height: 1.6;'>
                <span style='color: #ffd700; font-weight: bold; font-family: Orbitron;'>[ WARNING ]</span> <span style='color: #e2e8f0; font-weight: bold;'>OPEC+ SHIFT</span><br>
                <span style='color: #cbd5e1;'>Saudi Arabia leaning toward resuming gradual production increases in April 2026.</span>
            </div>
            """, unsafe_allow_html=True)

else:
    # --- COUNTRY DASHBOARD VIEW (NATIVE STREAMLIT LAYOUT WITH COLORS) ---
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
        st.markdown(f"<h4 style='color:#00f0ff; margin-bottom: 15px;'>VOLATILITY_IMPACT_MATRIX</h4>", unsafe_allow_html=True)
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        hex_color = intel['color'].lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgba_fill = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.1)'

        fig.add_trace(go.Scatter(x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Vol', line=dict(color=intel['color'], width=3, shape='spline'), fill='tozeroy', fillcolor=rgba_fill), secondary_y=False)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI ($)', line=dict(color="#475569", width=2, dash='dot')), secondary_y=True)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0")))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='rgba(0, 240, 255, 0.1)', secondary_y=False)
        fig.update_yaxes(title_text="WTI", color="#475569", showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # New Feature Importance Chart Below Volatility Matrix
        st.plotly_chart(create_feature_importance_chart(intel['color']), use_container_width=True, config={'displayModeBar': False})

    with col_right:
        # Pure Streamlit Container with COLORIZED HTML/Markdown text
        with st.container():
            # LOCAL DYNAMICS
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ LOCAL_DYNAMICS ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #cbd5e1; line-height: 1.6; padding-left: 12px; border-left: 2px solid #00f0ff;'>
                    {intel['info']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # MARKET DATA
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ MARKET_DATA ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'>
                    <span style='color: #00f0ff;'>&gt;</span> {intel['figures']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # EVENT IMPACT (Colored based on risk)
            st.markdown(f"""
            <div style='margin-bottom: 25px;'>
                <p style='color: {intel['color']}; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px; text-shadow: 0 0 8px {hex_to_rgba(intel['color'], 0.4)};'>[ EVENT_IMPACT ]</p>
                <p style='font-family: JetBrains Mono; font-size: 13px; color: #e2e8f0; line-height: 1.6;'>
                    <span style='color: {intel['color']};'>&gt;</span> {intel['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='border-top: 1px solid rgba(0, 240, 255, 0.3); margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            
            # SYSTEM FORECAST
            st.markdown(f"""
            <div>
                 <p style='color: #00f0ff; font-family: Orbitron; font-size: 14px; font-weight: bold; margin-bottom: 8px;'>[ SYSTEM_FORECAST ]</p>
                 <ul style='font-family: "JetBrains Mono"; font-size: 13px; line-height: 2.0; color: #94a3b8; list-style-type: none; padding-left: 0;'>
                    <li><span style='color: #00f0ff;'>[ GPR_TRACKING ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{(latest.get('gpr', 50)*mod):.1f}</span></li>
                    <li><span style='color: {intel["color"]};'>[ FORECAST ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{'Escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else ('Maintain' if intel['risk'] == 'MEDIUM' else 'Stabilize')}</span></li>
                    <li><span style='color: #00f0ff;'>[ CORRELATION ]</span> <span style='color: #e2e8f0; font-weight: bold;'>{'High' if intel['risk'] in ['MEDIUM', 'HIGH'] else 'Moderate'}</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
