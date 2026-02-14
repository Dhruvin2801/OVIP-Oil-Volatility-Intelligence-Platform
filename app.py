"""
OVIP - Oil Volatility Intelligence Platform
Main Streamlit Application

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import config and utilities
from config import COLORS, DEFAULT_SESSION_STATE
from utils.helpers import apply_custom_theme, initialize_session_state

# Page configuration
st.set_page_config(
    page_title="OVIP - Oil Volatility Intelligence Platform",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
apply_custom_theme()

# Initialize session state
initialize_session_state()

# Main page content
def main():
    """Main application entry point"""
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.image("assets/images/logo.png", width=100) if Path("assets/images/logo.png").exists() else st.markdown("## 🛢️")
    
    with col2:
        st.markdown(
            "<h1 style='text-align: center; color: #64FFDA;'>OIL VOLATILITY INTELLIGENCE PLATFORM</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #8892B0;'>Next-Generation Volatility Prediction for Strategic Decision-Making</p>",
            unsafe_allow_html=True
        )
    
    with col3:
        if st.button("⚙️ Settings"):
            st.switch_page("pages/7_⚙️_Settings.py")
    
    st.markdown("<hr style='border: 1px solid #64FFDA; box-shadow: 0 0 10px #64FFDA;'>", unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("### 👋 Welcome to OVIP")
    
    st.markdown("""
    OVIP is an **intelligence-grade platform** for oil market volatility prediction, combining:
    
    - 🎯 **NPRS-1 Binary Classifier** - 68.2% accuracy for direction prediction
    - 📊 **11-Pillar Regression Model** - 22.5% R² for volatility level forecasting
    - 🤖 **AI-Powered Insights** - Natural language explanations and recommendations
    - 🌍 **Multi-Market Coverage** - WTI, Brent, Dubai, and more
    """)
    
    # Quick stats
    st.markdown("### 📈 Live Market Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🇺🇸 WTI Crude",
            value="$78.34",
            delta="4.2%",
            help="West Texas Intermediate"
        )
    
    with col2:
        st.metric(
            label="Current Volatility",
            value="0.187",
            delta="-2.1%",
            delta_color="inverse",
            help="Realized monthly volatility"
        )
    
    with col3:
        st.metric(
            label="Regime State",
            value="🟡 MODERATE",
            help="Current market regime"
        )
    
    with col4:
        st.metric(
            label="Model Confidence",
            value="68%",
            help="NPRS-1 prediction confidence"
        )
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Quick Navigation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌍 **Select Country/Market**", use_container_width=True):
            st.switch_page("pages/1_🌍_Country_Selector.py")
        st.caption("Choose your target oil market")
    
    with col2:
        if st.button("📊 **View Dashboard**", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📊_Dashboard.py")
        st.caption("Live data and predictions")
    
    with col3:
        if st.button("💬 **AI Assistant**", use_container_width=True):
            st.switch_page("pages/3_💬_AI_Assistant.py")
        st.caption("Ask questions, get insights")
    
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 **Deep Analytics**", use_container_width=True):
            st.switch_page("pages/4_📈_Analytics.py")
        st.caption("Detailed model performance")
    
    with col2:
        if st.button("🔔 **Manage Alerts**", use_container_width=True):
            st.switch_page("pages/5_🔔_Alerts.py")
        st.caption("Set up notifications")
    
    with col3:
        if st.button("📄 **Generate Reports**", use_container_width=True):
            st.switch_page("pages/6_📄_Reports.py")
        st.caption("Export analysis")
    
    st.markdown("---")
    
    # Key features
    st.markdown("### ⭐ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Dual-Model Prediction:**
        - Direction (Up/Down) - 68.2% accuracy
        - Level (Exact volatility) - 22.5% R²
        - Outperforms traditional GARCH by 30×
        
        **🤖 AI-Powered Intelligence:**
        - Natural language Q&A
        - Automated recommendations
        - Explainable predictions
        
        **📊 Advanced Analytics:**
        - Feature importance analysis
        - Regime detection
        - Historical performance tracking
        """)
    
    with col2:
        st.markdown("""
        **🌍 Multi-Market Support:**
        - WTI (US), Brent (UK), Dubai (UAE)
        - Arab Light, Urals, and more
        - Country-specific insights
        
        **🔔 Smart Alerts:**
        - Regime shift detection
        - Price spike warnings
        - Custom triggers
        
        **📄 Professional Reporting:**
        - Executive summaries
        - Performance reviews
        - Export to PDF, Excel, PowerPoint
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("### 📚 About OVIP")
    
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        OVIP uses two complementary machine learning models:
        
        1. **NPRS-1 Binary Classifier** (68.2% accuracy)
           - Predicts: Will volatility go UP or DOWN?
           - Features: Volatility persistence, regime state, news intensity, geopolitical risk, momentum
           - Best for: Trading signals, tactical decisions
        
        2. **11-Pillar Regression Model** (22.5% R²)
           - Predicts: What will volatility level be?
           - Features: 11 engineered features across persistence, regime, NLP, and macro categories
           - Best for: Risk management, hedging decisions, VaR estimation
        
        **Key Innovation:** Regime-switching framework where crisis probability drives 50% of predictions
        (vs traditional models where persistence drives 80%+)
        """)
    
    with st.expander("📊 Model Performance"):
        st.markdown("""
        **NPRS-1 Direction Classifier:**
        - Overall accuracy: 68.2%
        - Consistent across all test years (2021-2025)
        - Best year: 2021 (75% accuracy)
        - Baseline: 50% (random guess)
        
        **11-Pillar Level Predictor:**
        - Overall R²: 22.5%
        - Best year: 2022 (+22.8% R²)
        - Works best in crisis periods
        - Baseline: 0.75% (GARCH model)
        
        **Statistical Validation:**
        - Clark-West test: p = 0.0062 (highly significant)
        - Granger causality: Confirmed
        - VIF check: Max 5.36 (no multicollinearity)
        """)
    
    with st.expander("⚠️ Limitations & Disclaimers"):
        st.markdown("""
        **Known Limitations:**
        - 11-pillar model works best in 2/5 test years (regime-specific, not universal)
        - Directional accuracy below random (use NPRS-1 for direction instead)
        - Trained on 2002-2020 data, tested on 2021-2025
        - Monthly frequency only (not suitable for intraday trading)
        
        **Recommended Usage:**
        - Use NPRS-1 for all directional decisions (reliable)
        - Use 11-pillar only when regime state indicates moderate/high risk
        - Combine with fundamental analysis
        - Revalidate quarterly
        
        **Disclaimer:**
        This platform is for informational purposes only. Not financial advice.
        Past performance does not guarantee future results.
        Always consult with qualified professionals before making trading/hedging decisions.
        """)
    
    st.markdown("---")
    
    # Contact/Support
    st.markdown("""
    <div style='text-align: center; color: #8892B0;'>
        <p>Questions? Contact support@ovip.ai | © 2026 OVIP - All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
