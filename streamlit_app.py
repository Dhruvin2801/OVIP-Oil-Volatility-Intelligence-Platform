import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Safely import AI engine
try:
    from modules.ai_engine import setup_rag_vector_db, get_ai_response
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(page_title="OVIP // Analytics Platform", layout="wide", initial_sidebar_state="expanded")

# Minimal CSS just to tighten up padding (No aggressive locks)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-weight: 600; }
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
    
    # Failsafe dummy data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150), 'gpr': np.random.rand(150)*100})

df_main = load_data()

vec, tfidf, rag_df = None, None, None
if AI_AVAILABLE:
    try: vec, tfidf, rag_df = setup_rag_vector_db(df_main)
    except: pass

# ==========================================
# 3. GLOBAL NODE DATABASE (30 Countries)
# ==========================================
if 'target' not in st.session_state: st.session_state.target = None

COUNTRIES = {
    'INDIA': {'lat': 20.59, 'lon': 78.96, 'risk': 'LOW', 'event': 'Refinery intake stable. Russian crude discount shrinking.', 'color': '#2ca02c', 'mod': 0.95},
    'USA': {'lat': 37.09, 'lon': -95.71, 'risk': 'MEDIUM', 'event': 'SPR refill operations ongoing. Shale production flat.', 'color': '#1f77b4', 'mod': 1.0},
    'CHINA': {'lat': 35.86, 'lon': 104.20, 'risk': 'HIGH', 'event': 'Industrial demand slowdown. Import quotas tightening.', 'color': '#d62728', 'mod': 1.1},
    'RUSSIA': {'lat': 61.52, 'lon': 105.31, 'risk': 'CRITICAL', 'event': 'Sanction evasion routing detected via dark fleet.', 'color': '#d62728', 'mod': 1.25},
    'SAUDI ARABIA': {'lat': 23.89, 'lon': 45.08, 'risk': 'MEDIUM', 'event': 'OPEC+ voluntary cuts maintained. Spare capacity high.', 'color': '#ff7f0e', 'mod': 1.0},
    'UAE': {'lat': 23.42, 'lon': 53.85, 'risk': 'LOW', 'event': 'Fujairah bunkering volumes spike. Storage expanding.', 'color': '#2ca02c', 'mod': 0.98},
    'IRAN': {'lat': 32.42, 'lon': 53.68, 'risk': 'CRITICAL', 'event': 'Strait of Hormuz transit harassment reported.', 'color': '#d62728', 'mod': 1.3},
    'VENEZUELA': {'lat': 6.42, 'lon': -66.58, 'risk': 'HIGH', 'event': 'Production stagnating due to lack of capital expenditure.', 'color': '#d62728', 'mod': 1.15},
    'BRAZIL': {'lat': -14.23, 'lon': -51.92, 'risk': 'LOW', 'event': 'Pre-salt offshore production hitting record highs.', 'color': '#2ca02c', 'mod': 0.9},
    'UK': {'lat': 55.37, 'lon': -3.43, 'risk': 'MEDIUM', 'event': 'North Sea windfall tax impacts future drilling plans.', 'color': '#1f77b4', 'mod': 1.02},
    'NORWAY': {'lat': 60.47, 'lon': 8.46, 'risk': 'LOW', 'event': 'Equinor gas exports to Europe maxed. Secure supply.', 'color': '#2ca02c', 'mod': 0.9},
    'NIGERIA': {'lat': 9.08, 'lon': 8.67, 'risk': 'HIGH', 'event': 'Pipeline vandalism causes force majeure on Bonny Light.', 'color': '#d62728', 'mod': 1.2},
    'ANGOLA': {'lat': -11.20, 'lon': 17.87, 'risk': 'MEDIUM', 'event': 'Deepwater asset decline rates accelerating.', 'color': '#ff7f0e', 'mod': 1.05},
    'LIBYA': {'lat': 26.33, 'lon': 17.22, 'risk': 'CRITICAL', 'event': 'Eastern faction blockades export terminals.', 'color': '#d62728', 'mod': 1.4},
    'IRAQ': {'lat': 33.22, 'lon': 43.67, 'risk': 'HIGH', 'event': 'Federal vs Kurdish export disputes unresolved.', 'color': '#d62728', 'mod': 1.15},
    'KUWAIT': {'lat': 29.31, 'lon': 47.48, 'risk': 'LOW', 'event': 'Al-Zour refinery running at full capacity.', 'color': '#2ca02c', 'mod': 0.95},
    'QATAR': {'lat': 25.35, 'lon': 51.18, 'risk': 'LOW', 'event': 'North Field LNG expansion on schedule.', 'color': '#2ca02c', 'mod': 0.95},
    'CANADA': {'lat': 56.13, 'lon': -106.34, 'risk': 'LOW', 'event': 'Trans Mountain pipeline expansion easing bottlenecks.', 'color': '#2ca02c', 'mod': 0.98},
    'MEXICO': {'lat': 23.63, 'lon': -102.55, 'risk': 'MEDIUM', 'event': 'Pemex debt restructuring causing operational delays.', 'color': '#ff7f0e', 'mod': 1.05},
    'GERMANY': {'lat': 51.16, 'lon': 10.45, 'risk': 'MEDIUM', 'event': 'Macro slowdown reducing middle-distillate demand.', 'color': '#1f77b4', 'mod': 1.0},
    'JAPAN': {'lat': 36.20, 'lon': 138.25, 'risk': 'MEDIUM', 'event': 'Nuclear restart program slowing LNG imports slightly.', 'color': '#1f77b4', 'mod': 1.0},
    'SOUTH KOREA': {'lat': 35.90, 'lon': 127.76, 'risk': 'LOW', 'event': 'Strategic reserves fully stocked.', 'color': '#2ca02c', 'mod': 0.95},
    'AUSTRALIA': {'lat': -25.27, 'lon': 133.77, 'risk': 'LOW', 'event': 'Gorgon LNG facility maintenance completed.', 'color': '#2ca02c', 'mod': 0.95},
    'ALGERIA': {'lat': 28.03, 'lon': 1.65, 'risk': 'MEDIUM', 'event': 'European gas supply contracts renegotiated.', 'color': '#ff7f0e', 'mod': 1.0},
    'EGYPT': {'lat': 26.82, 'lon': 30.80, 'risk': 'HIGH', 'event': 'Suez Canal transit risks remain elevated due to regional conflict.', 'color': '#d62728', 'mod': 1.15},
    'TURKEY': {'lat': 38.96, 'lon': 35.24, 'risk': 'MEDIUM', 'event': 'Bosphorus transit fees adjusted. Russian imports high.', 'color': '#ff7f0e', 'mod': 1.05},
    'SOUTH AFRICA': {'lat': -30.55, 'lon': 22.93, 'risk': 'MEDIUM', 'event': 'Cape of Good Hope rerouting adding 14 days to freight.', 'color': '#ff7f0e', 'mod': 1.1},
    'SINGAPORE': {'lat': 1.35, 'lon': 103.81, 'risk': 'LOW', 'event': 'Malacca Strait maritime traffic flowing normally.', 'color': '#2ca02c', 'mod': 0.95},
    'INDONESIA': {'lat': -0.78, 'lon': 113.92, 'risk': 'MEDIUM', 'event': 'Domestic subsidy policies straining fiscal budget.', 'color': '#ff7f0e', 'mod': 1.0},
    'OMAN': {'lat': 21.51, 'lon': 55.92, 'risk': 'LOW', 'event': 'Gulf of Oman crude blending operations expanding.', 'color': '#2ca02c', 'mod': 0.95}
}

# ==========================================
# 4. SIDEBAR: AI COPILOT
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Globe_icon.svg", width=50) # simple logo
    st.title("OVIP Daemon")
    st.markdown("---")
    
    # Initialize chat history
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [{"role": "assistant", "content": "Welcome. Select a country on the map or ask a macro question."}]

    # Render chat messages natively
    for msg in st.session_state.chat_log:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Native Chat Input (Always anchored at bottom of sidebar)
    if prompt := st.chat_input("Ask about market trends..."):
        st.session_state.chat_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if AI_AVAILABLE and vec is not None:
                    context_prompt = f"Target Region: {st.session_state.target if st.session_state.target else 'Global'}. Query: {prompt}"
                    ans = get_ai_response(context_prompt, vec, tfidf, rag_df)
                else: 
                    ans = "⚠️ AI Engine Offline. Please check API keys or vector DB."
                st.markdown(ans)
        st.session_state.chat_log.append({"role": "assistant", "content": ans})

# ==========================================
# 5. MAIN VIEW: GLOBE OR DASHBOARD
# ==========================================
if st.session_state.target is None:
    # --- GLOBE VIEW ---
    st.title("Global Threat Matrix")
    st.markdown("Select a strategic node to view localized volatility impact and geopolitical risk.")
    
    lats = [v['lat'] for v in COUNTRIES.values()]
    lons = [v['lon'] for v in COUNTRIES.values()]
    names = list(COUNTRIES.keys())
    colors = [v['color'] for v in COUNTRIES.values()]
    
    fig_globe = go.Figure(go.Scattergeo(
        lon = lons, lat = lats, text = names, mode = 'markers+text',
        marker = dict(size=12, color=colors, line=dict(width=1, color='#ffffff')),
        textfont = dict(family="Arial", size=11, color="#ffffff"),
        textposition = "top center"
    ))
    
    # Clean, modern globe facing India
    fig_globe.update_geos(
        projection_type="orthographic", showcoastlines=True, coastlinecolor="#444444",
        showland=True, landcolor="#1e1e1e", showocean=True, oceancolor="#0e1117",
        showcountries=True, countrycolor="#333333", bgcolor="rgba(0,0,0,0)",
        center=dict(lon=78.96, lat=20.59), projection_rotation=dict(lon=78.96, lat=20.59, roll=0)
    )
    fig_globe.update_layout(height=700, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    # Render Interactive Globe
    event = st.plotly_chart(fig_globe, on_select="rerun", selection_mode="points", use_container_width=True)
    
    # Handle click event
    if event and "selection" in event and event["selection"]["points"]:
        st.session_state.target = names[event["selection"]["points"][0]["point_index"]]
        st.rerun()

else:
    # --- COUNTRY DASHBOARD VIEW ---
    target = st.session_state.target
    intel = COUNTRIES[target]
    latest = df_main.iloc[-1]
    mod = intel['mod']
    
    # Header with Back Button
    col_title, col_btn = st.columns([5, 1])
    col_title.title(f"{target} Analytics")
    if col_btn.button("← Return to Globe", use_container_width=True):
        st.session_state.target = None
        st.rerun()

    # Metrics Row
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regional WTI Premium", f"${(latest.get('WTI', 75) * mod):.2f}", f"{'High Risk' if intel['risk'] in ['HIGH', 'CRITICAL'] else 'Stable'}")
    m2.metric("Local Volatility (Sigma)", f"{(latest.get('Volatility', 0.1) * mod):.3f}")
    m3.metric("Geopolitical Risk (GPR)", f"{(latest.get('gpr', 50) * mod):.1f}")
    m4.metric("Threat Level", intel['risk'])
    st.markdown("---")

    # Content Row: Chart & Intelligence
    col_chart, col_intel = st.columns([2.5, 1])

    with col_chart:
        st.subheader("Volatility Impact Matrix")
        chart_df = df_main.dropna(subset=['Date']).tail(100).copy()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Localized Volatility (Color coded to country risk)
        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=(chart_df['Volatility'] * mod), name=f'{target} Volatility',
            line=dict(color=intel['color'], width=3), fill='tozeroy', fillcolor=intel['color'].replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in intel['color'] else None # Simplistic fill
        ), secondary_y=False)

        # Global WTI
        fig.add_trace(go.Scatter(
            x=chart_df['Date'], y=chart_df['WTI'], name='Global WTI Price ($)',
            line=dict(color="#888888", width=2, dash='dot')
        ), secondary_y=True)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor='#333333')
        fig.update_yaxes(title_text="Volatility", color=intel['color'], showgrid=True, gridcolor='#333333', secondary_y=False)
        fig.update_yaxes(title_text="WTI Price", color="#888888", showgrid=False, secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with col_intel:
        st.subheader("Regional Intelligence")
        
        # Display Country Specific Event in a nice native info box
        if intel['risk'] in ['HIGH', 'CRITICAL']:
            st.error(f"**Alert:** {intel['event']}")
        elif intel['risk'] == 'MEDIUM':
            st.warning(f"**Notice:** {intel['event']}")
        else:
            st.success(f"**Status:** {intel['event']}")
            
        st.write("")
        st.markdown(f"**Macro Insight:** Localized GPR index tracking at **{(latest.get('gpr', 50)*mod):.1f}**. Volatility is expected to **{'escalate' if intel['risk'] in ['HIGH', 'CRITICAL'] else 'stabilize'}** over the next 72 hours.")
        
        st.write("")
        st.info("💡 **Tip:** Use the AI Copilot in the sidebar to ask specific questions about how this region impacts the global supply chain.")
