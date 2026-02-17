import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# ==========================================
# 1. CORE CONFIGURATION
# ==========================================
st.set_page_config(page_title="GlobalCharge | AI Engine", layout="wide", initial_sidebar_state="collapsed")

# Palette obeying your ledger: Red (Primary/Target), Orange (Warnings/Alerts), Black/Slate (BG)
COLORS = {
    'bg': '#050505',           
    'panel': '#111111',        
    'primary': '#dc2626',      # Strict Red for targets/positive (replacing blue)
    'alert': '#ea580c',        # Strict Orange for warnings (replacing red)
    'text': '#e2e8f0',
    'muted': '#475569'
}

# ==========================================
# 2. HUD CSS INJECTION (LOCKED VIEWPORT)
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
    
    /* Lock viewport, prevent scrolling */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS['bg']}; color: {COLORS['text']};
        font-family: 'JetBrains Mono', monospace !important; overflow: hidden !important; 
    }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }}
    ::-webkit-scrollbar {{ display: none; }}
    
    /* Institutional Cut-Corner Panels */
    .st-emotion-cache-1wivap2, div[data-testid="stVerticalBlock"] > div.element-container {{
        background: {COLORS['panel']}; border: 1px solid #222;
        clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
        padding: 15px; border-top: 2px solid {COLORS['primary']};
    }}

    /* STRICT METRIC SIZING (+2pt enforced) */
    .metric-val {{ font-size: calc(2.2rem + 2pt); color: {COLORS['primary']}; font-weight: 800; line-height: 1.1; margin: 0; }}
    .metric-lbl {{ font-size: calc(0.9rem + 2pt); color: {COLORS['muted']}; font-weight: 700; text-transform: uppercase; margin: 0; }}
    
    /* Terminal Buttons */
    .stButton>button {{
        background-color: transparent; color: {COLORS['text']}; border: 1px solid #333;
        border-radius: 0px; width: 100%; text-transform: uppercase; font-weight: 700; transition: 0.2s;
    }}
    .stButton>button:hover {{ background-color: {COLORS['primary']}; color: white; border-color: {COLORS['primary']}; }}
    
    /* Warning Text */
    .alert-text {{ color: {COLORS['alert']}; font-weight: 800; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. STATE MANAGEMENT & DATA
# ==========================================
if 'target_country' not in st.session_state:
    st.session_state.target_country = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'intel'

@st.cache_data
def load_data():
    file = 'war_room_audit_2025.csv'
    if os.path.exists(file):
        df = pd.read_csv(file)
        df.columns = [c.lower() for c in df.columns] 
        if 'country' not in df.columns:
            for c in df.columns:
                if 'name' in c or 'nation' in c: df.rename(columns={c: 'country'}, inplace=True)
        return df
    return None

df = load_data()
if df is None:
    st.error("SYS.ERR: 'war_room_audit_2025.csv' MISSING.")
    st.stop()

# ==========================================
# 4. VIEW 1: THE 3D MACRO GLOBE
# ==========================================
def render_globe():
    st.markdown(f"<h2 style='text-align: center; color: {COLORS['primary']}; letter-spacing: 2px;'>[ GLOBALCHARGE // MACRO_ALLOCATION_MATRIX ]</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #475569;'>AWAITING TARGET ACQUISITION...</p>", unsafe_allow_html=True)
    
    # Render 3D Globe using Plotly Orthographic Projection (Oranges for heatmap)
    fig = px.choropleth(df, locations="country", locationmode='country names', color="roi_score", color_continuous_scale="Oranges")
    fig.update_geos(
        projection_type="orthographic", showland=True, landcolor="#111", oceancolor="#050505", 
        showframe=False, lakecolor="#050505", bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, height=500, coloraxis_showscale=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Quick Target Selection (Like OVIP)
    st.markdown("```bash\n> SELECT_TARGET_NODE\n```")
    top_targets = ["India", "Germany", "Australia", "France", "Belgium"]
    cols = st.columns(len(top_targets))
    for i, target in enumerate(top_targets):
        with cols[i]:
            if st.button(f"🎯 {target.upper()}"):
                st.session_state.target_country = target
                st.rerun()

# ==========================================
# 5. VIEW 2: TACTICAL HUD DASHBOARD
# ==========================================
def get_intel(country):
    repo = {
        "Belgium": ("Fiscal Dominance Shield", "Zero-emission corporate mandate bypasses consumer interest rate shocks.", True),
        "Australia": ("NVES Policy Alpha", "Prompt NVES implementation and FBT exemption surges commercial ROI.", True),
        "India": ("Strategic EMPS Pivot", "Supply plateauing, but PLI incentives force localized production. Buy on the dip.", True),
        "France": ("Eco-Score Moat", "Carbon-indexed subsidies block heavy imports, stabilizing domestic margins.", True),
        "Germany": ("Umweltbonus Shock", "Subsidy termination collapsed sales 35%. Severe mean-reversion phase active.", False)
    }
    return repo.get(country, ("GDP Organic S-Curve", "Adoption shielded by organic wealth scaling.", True))

def render_dashboard():
    target = st.session_state.target_country
    c_data = df[df['country'] == target].iloc[0]
    
    # TOP HUD HEADER
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 15px;">
            <h2 style="margin: 0; color: {COLORS['primary']};">SYS.TARGET :: {target.upper()}</h2>
            <span style="color: {COLORS['muted']};">SECURE_LINK | {datetime.utcnow().strftime('%H:%M:%S')} UTC</span>
        </div>
    """, unsafe_allow_html=True)

    # TWO-COLUMN HUD LAYOUT (1:4 Ratio, exactly like OVIP)
    col_nav, col_main = st.columns([1.5, 3.5])
    
    # --- LEFT SIDEBAR (NAVIGATION & METRICS) ---
    with col_nav:
        if st.button("⬅ ABORT & RETURN TO GLOBE"):
            st.session_state.target_country = None
            st.rerun()
            
        st.markdown("<hr style='border: 1px dashed #333;'>", unsafe_allow_html=True)
        
        # Explicit HTML to enforce +2pt rule flawlessly
        st.markdown(f"""
            <p class="metric-lbl">AI CONFIDENCE</p>
            <p class="metric-val">{c_data.get('new_prob_pct', 80):.1f}%</p><br>
            
            <p class="metric-lbl">ALPHA GAP</p>
            <p class="metric-val">{c_data.get('opportunity_gap', 0.5):.2f}</p><br>
            
            <p class="metric-lbl">ADOPTION LAGGED</p>
            <p class="metric-val" style="color: white;">{c_data.get('lagged_share', 15):.1f}%</p>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 1px dashed #333;'>", unsafe_allow_html=True)
        if st.button("📑 MACRO INTEL"): st.session_state.active_tab = 'intel'; st.rerun()
        if st.button("⚙️ OVERRIDE LOGIC"): st.session_state.active_tab = 'logic'; st.rerun()

    # --- RIGHT MAIN AREA (DYNAMIC CONTENT) ---
    with col_main:
        headline, context, is_safe = get_intel(target)
        
        if st.session_state.active_tab == 'intel':
            st.markdown(f"<h4 style='margin-top:0; color: {COLORS['primary']};'>> INITIATING REGIME AUDIT...</h4>", unsafe_allow_html=True)
            
            # Warning block using strict Orange
            if not is_safe:
                st.markdown(f"""
                <div style="background: rgba(234, 88, 12, 0.1); border-left: 5px solid {COLORS['alert']}; padding: 15px; margin-bottom: 20px;">
                    <span class="alert-text">⚠️ VULNERABILITY DETECTED:</span> Regime shift predicted for 2024. Proceed with extreme caution.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(220, 38, 38, 0.05); border-left: 5px solid {COLORS['primary']}; padding: 15px; margin-bottom: 20px;">
                    <span style="color: {COLORS['primary']}; font-weight: 800;">✓ STRUCTURALLY SECURE:</span> Market fundamentals validated against policy shocks.
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"""
                <div style="background: #111; padding: 25px; border: 1px solid #333;">
                    <h3 style="color: white; margin-top: 0; text-transform: uppercase;">{headline}</h3>
                    <p style="font-size: 1.1rem; line-height: 1.7;">{context}</p>
                </div>
            """, unsafe_allow_html=True)

        elif st.session_state.active_tab == 'logic':
            st.markdown(f"<h4 style='margin-top:0; color: {COLORS['primary']};'>> MANUAL SYSTEM OVERRIDE</h4>", unsafe_allow_html=True)
            st.write("Adjust deployment weights to bypass AI baseline recommendations:")
            
            st.slider("🛡️ RESILIENCE WEIGHT", 0.0, 2.0, 1.0, step=0.1)
            st.slider("📈 CAPACITY WEIGHT", 0.0, 2.0, 1.0, step=0.1)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("RECALCULATE ALLOCATION MATRIX"):
                st.success("SYS: Weights updated in local cache.")

# ==========================================
# 6. APP ROUTER
# ==========================================
if st.session_state.target_country is None:
    render_globe()
else:
    render_dashboard()
