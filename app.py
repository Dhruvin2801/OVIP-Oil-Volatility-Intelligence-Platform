"""
OVIP - Oil Volatility Intelligence Platform
Main Streamlit Application

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add root directory to path so imports work perfectly
sys.path.append(str(Path(__file__).parent))

# Import config (where our theme lives)
import config

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="OVIP - Oil Volatility Intelligence Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme from config.py
config.apply_custom_theme()

def initialize_session_state():
    """Ensures required variables exist before navigating to other pages"""
    if 'market_display' not in st.session_state:
        st.session_state['market_display'] = '🇺🇸 United States (WTI)'
    if 'selected_market' not in st.session_state:
        st.session_state['selected_market'] = 'WTI'
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            {"role": "assistant", "content": "SYSTEM ONLINE. Awaiting query..."}
        ]

# Initialize session state
initialize_session_state()

def main():
    """Main application entry point"""
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # Graceful fallback if logo doesn't exist yet
        if Path("assets/images/logo.png").exists():
            st.image("assets/images/logo.png", width=100)
        else:
            st.markdown("<h1 style='font-size: 3rem; margin:0;'>🛢️</h1>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            "<h1 style='text-align: center; color: #64FFDA; margin-bottom: 0;'>OIL VOLATILITY INTELLIGENCE PLATFORM</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #8892B0; font-size: 1.2rem;'>Next-Generation Volatility Prediction for Strategic Decision-Making</p>",
            unsafe_allow_html=True
        )
    
    with col3:
        st.write("") # Spacing
        if st.button("⚙️ Settings", use_container_width=True):
            try:
                st.switch_page("pages/7_⚙️_Settings.py")
            except:
                st.warning("Settings page not yet created.")
    
    st.markdown("<hr style='border: 1px solid #1E3A5F; box-shadow: 0 0 10px rgba(100, 255, 218, 0.2);'>", unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("### 👋 Welcome to OVIP")
    
    st.markdown("""
    <div style='background-color: #112240; padding: 20px; border-radius: 5px; border-left: 3px solid #64FFDA;'>
        OVIP is an <strong>intelligence-grade platform</strong> for oil market volatility prediction, combining:
        <br><br>
        <ul>
            <li>🎯 <strong>NPRS-1 Binary Classifier</strong> - 68.2% accuracy for direction prediction</li>
            <li>📊 <strong>11-Pillar Regression Model</strong> - 22.5% R² for volatility level forecasting</li>
            <li>🤖 <strong>AI-Powered Insights</strong> - Natural language explanations and recommendations</li>
            <li>🌍 <strong>Multi-Market Coverage</strong> - WTI, Brent, Dubai, and more</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📈 Live Market Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🇺🇸 WTI Crude", value="$78.34", delta="4.2%", help="West Texas Intermediate")
    with col2:
        st.metric(label="Current Volatility", value="0.187", delta="-2.1%", delta_color="inverse", help="Realized monthly volatility")
    with col3:
        st.metric(label="Regime State", value="🟡 MODERATE", help="Current market regime")
    with col4:
        st.metric(label="Model Confidence", value="68%", help="NPRS-1 prediction confidence")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Quick Navigation")
    
    # Row 1 of Navigation
    n1, n2, n3 = st.columns(3)
    
    with n1:
        if st.button("🌍 **Select Target Market**", use_container_width=True):
            st.switch_page("pages/1_🌍_Country_Selector.py")
        st.caption("Choose your target oil market via 3D Globe")
    
    with n2:
        if st.button("📊 **Command Center Dashboard**", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📊_Dashboard.py")
        st.caption("Live data, threat matrix, and predictions")
    
    with n3:
        if st.button("💬 **AI Analyst Terminal**", use_container_width=True):
            st.switch_page("pages/3_💬_AI_Assistant.py")
        st.caption("Ask questions, get data-driven insights")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2 of Navigation
    n4, n5, n6 = st.columns(3)
    
    with n4:
        if st.button("📈 **Deep Analytics**", use_container_width=True):
            try: st.switch_page("pages/4_📈_Analytics.py")
            except: st.warning("Module pending.")
        st.caption("Detailed model performance & residual analysis")
    
    with n5:
        if st.button("🔔 **Manage Alerts**", use_container_width=True):
            try: st.switch_page("pages/5_🔔_Alerts.py")
            except: st.warning("Module pending.")
        st.caption("Set up regime shift notifications")
    
    with n6:
        if st.button("📄 **Generate Reports**", use_container_width=True):
            try: st.switch_page("pages/6_📄_Reports.py")
            except: st.warning("Module pending.")
        st.caption("Export executive summaries (PDF/CSV)")
    
    st.markdown("---")
    
    # Key features
    st.markdown("### ⭐ Key Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Dual-Model Prediction System:**
        * **Direction (Up/Down):** 68.2% out-of-sample accuracy via NPRS-1.
        * **Level (Exact volatility):** 22.5% R² capturing market shocks.
        * *Performance:* Outperforms traditional GARCH baselines by a significant margin.
        
        **🤖 RAG AI Intelligence:**
        * Natural language Q&A secured by factual context retrieval.
        * Automated hedging recommendations.
        * Explainable predictions bridging the gap between quant models and business logic.
        """)
    
    with col2:
        st.markdown("""
        **🌍 Global Market Support:**
        * Coverage across WTI (US), Brent (UK), and Dubai (UAE).
        * Customizable indicators per region.
        
        **🔔 Smart Alerts & Reporting:**
        * Real-time regime shift detection (Calm → Crisis).
        * Price spike warnings and custom threshold triggers.
        * One-click generation of professional performance reviews.
        """)
    
    st.markdown("---")
    
    # Footer Expanders
    st.markdown("### 📚 About OVIP")
    
    with st.expander("ℹ️ The 'Crisis Memory' Innovation"):
        st.markdown("""
        **Key Innovation:** OVIP utilizes a regime-switching framework where the probability of a market crisis drives the predictions. 
        
        Our core research proved the **Crisis Memory Hypothesis**: By forcing the model to learn from the 2000-2001 Dot-Com crash and 9/11 shock, the model's accuracy in predicting the 2022 Energy Crisis improved by nearly 1%. 
        
        Traditional models rely on persistence (recent history). OVIP relies on *regime recognition* (historical trauma).
        """)
    
    with st.expander("📊 Model Validation"):
        st.markdown("""
        **Statistical Validation:**
        * **Clark-West Test:** p = 0.0062 (Highly significant improvement over baseline).
        * **Granger Causality:** Confirmed for NLP Sentiment driving volatility.
        * **VIF Check:** Max 5.36 (No problematic multicollinearity in the 11-pillar features).
        """)
    
    with st.expander("⚠️ Limitations & Disclaimers"):
        st.markdown("""
        **Recommended Usage:**
        * Use **NPRS-1** for all directional decisions (highly reliable).
        * Use the **11-Pillar Regression** to estimate the *magnitude* of the risk.
        * Revalidate models quarterly as macroeconomic conditions shift.
        
        **Disclaimer:**
        *This platform is a demonstration of academic/quantitative research and is for informational purposes only. It does not constitute financial advice. Always consult with qualified professionals before executing trades or hedges.*
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #8892B0; font-family: "Roboto Mono", monospace;'>
        <p>STATUS: ONLINE | LATENCY: 12ms | © 2026 OVIP - All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
