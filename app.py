"""
OVIP - Oil Volatility Intelligence Platform
Main Streamlit Application
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
        st.write("") 
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
        OVIP is an <strong>intelligence-grade platform</strong> for oil market volatility prediction.
        <ul>
            <li>🎯 <strong>NPRS-1 Binary Classifier</strong> - 68.2% accuracy</li>
            <li>📊 <strong>11-Pillar Regression Model</strong> - 22.5% R²</li>
            <li>🤖 <strong>AI-Powered Insights</strong> - Natural language recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("### 📈 Live Market Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="🇺🇸 WTI Crude", value="$78.34", delta="4.2%")
    with col2: st.metric(label="Current Volatility", value="0.187", delta="-2.1%", delta_color="inverse")
    with col3: st.metric(label="Regime State", value="🟡 MODERATE")
    with col4: st.metric(label="Model Confidence", value="68%")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Quick Navigation")
    n1, n2, n3 = st.columns(3)
    
    with n1:
        if st.button("🌍 **Select Target Market**", use_container_width=True):
            st.switch_page("pages/1_🌍_Country_Selector.py")
    
    with n2:
        if st.button("📊 **Command Center Dashboard**", use_container_width=True, type="primary"):
            st.switch_page("pages/2_📊_Dashboard.py")
    
    with n3:
        # Fixed path for the Assistant page
        if st.button("💬 **AI Analyst Terminal**", use_container_width=True):
            st.switch_page("pages/3_💬_AI_Assistant.py")
    
    # Row 2
    n4, n5, n6 = st.columns(3)
    with n4:
        if st.button("📈 **Deep Analytics**", use_container_width=True):
            try: st.switch_page("pages/4_📈_Analytics.py")
            except: st.warning("Module pending.")
    with n5:
        if st.button("🔔 **Manage Alerts**", use_container_width=True):
            try: st.switch_page("pages/5_🔔_Alerts.py")
            except: st.warning("Module pending.")
    with n6:
        if st.button("📄 **Generate Reports**", use_container_width=True):
            try: st.switch_page("pages/6_📄_Reports.py")
            except: st.warning("Module pending.")

    st.markdown("---")
    
    # Expanders & Footer
    with st.expander("ℹ️ The 'Crisis Memory' Innovation"):
        st.write("OVIP utilizes a regime-switching framework...")
    
    st.markdown("""
    <div style='text-align: center; color: #8892B0; font-family: "Roboto Mono", monospace;'>
        <p>STATUS: ONLINE | © 2026 OVIP</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
