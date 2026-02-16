import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="OVIP - Alerts", layout="wide")
config.apply_custom_theme()

st.markdown("<h2>🔔 OPERATIONAL TRIGGERS</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

if 'active_alerts' not in st.session_state:
    st.session_state.active_alerts = []

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### CONFIGURE NEW ALERT")
    with st.form("alert_form"):
        trigger_type = st.selectbox("Telemetry Source:", ["Regime Probability", "Volatility Level", "NLP Sentiment Score"])
        condition = st.selectbox("Condition:", ["Greater Than (>)", "Less Than (<)"])
        threshold = st.number_input("Threshold Value:", value=0.70, format="%.2f")
        
        if st.form_submit_button("DEPLOY TRIGGER"):
            st.session_state.active_alerts.append({
                "type": trigger_type, "cond": condition, "val": threshold, "time": datetime.utcnow().strftime('%H:%M:%S UTC')
            })
            st.success("Trigger deployed to memory.")

with c2:
    st.markdown("### ACTIVE WATCHLIST")
    if not st.session_state.active_alerts:
        st.info("No active triggers.")
    else:
        for alert in st.session_state.active_alerts:
            st.markdown(f"""
            <div style='background: {config.COLORS['surface']}; padding: 10px; border-left: 3px solid {config.COLORS['accent_primary']}; margin-bottom: 5px;'>
                <strong>{alert['type']}</strong> {alert['cond']} {alert['val']} <br>
                <small style='color: {config.COLORS['text_secondary']}'>Deployed: {alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)
        if st.button("CLEAR ALL TRIGGERS"):
            st.session_state.active_alerts = []
            st.rerun()
