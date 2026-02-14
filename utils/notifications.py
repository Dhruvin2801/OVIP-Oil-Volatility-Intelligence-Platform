"""
OVIP - Notifications Utility
Manages system-wide alerts and threshold triggers.
"""
import streamlit as st
from datetime import datetime
import uuid

def add_alert(title: str, message: str, severity: str = "INFO"):
    """
    Creates a new system alert and stores it globally.
    Severity levels: 'INFO', 'WARNING', 'CRITICAL'
    """
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
        
    alert = {
        'id': str(uuid.uuid4()),
        'title': title,
        'message': message,
        'severity': severity.upper(),
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'read': False
    }
    
    # Insert at the top of the list so the newest is first
    st.session_state.alerts.insert(0, alert)

def get_active_alerts():
    """Returns all unread alerts."""
    if 'alerts' not in st.session_state:
        return []
    return [a for a in st.session_state.alerts if not a['read']]

def dismiss_alert(alert_id: str):
    """Marks a specific alert as read."""
    if 'alerts' in st.session_state:
        for alert in st.session_state.alerts:
            if alert['id'] == alert_id:
                alert['read'] = True
                break

def check_market_thresholds(current_vol: float, prev_vol: float, regime_prob: float):
    """
    Business logic to automatically trigger alerts based on market telemetry.
    Can be called inside your dashboard data loading sequence.
    """
    # 1. Regime Shift Alert
    if regime_prob > 0.5:
        add_alert(
            title="⚠️ Regime Shift Detected", 
            message="Market has entered a CRISIS regime. High volatility expected.",
            severity="CRITICAL"
        )
    
    # 2. Volatility Spike Alert (e.g., > 20% jump)
    if prev_vol > 0 and ((current_vol - prev_vol) / prev_vol) > 0.20:
        add_alert(
            title="📈 Volatility Spike",
            message=f"Volatility has surged rapidly compared to previous reading.",
            severity="WARNING"
        )
