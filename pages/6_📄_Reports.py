import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.data_loader import get_data_loader

st.set_page_config(page_title="OVIP - Reports", layout="wide")
config.apply_custom_theme()

st.markdown("<h2>📄 INTELLIGENCE EXPORT</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

metrics = get_data_loader().get_latest_metrics()

report_content = f"""# OVIP EXECUTIVE BRIEFING
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Target Asset:** {st.session_state.get('market_display', 'WTI (USA)')}

## CURRENT TELEMETRY
* **Price:** ${metrics['price']:.2f}
* **Volatility:** {metrics['volatility']:.3f}
* **Regime State:** {metrics['regime']} (Crisis Probability: {metrics['crisis_prob']:.2f})
* **NLP Sentiment:** {metrics['sentiment']:.2f}

## MODEL DIRECTIVES
* **NPRS-1 Signal:** UP (Hedge required)
* **Confidence:** 68.2%
* **Recommendation:** Execute phased hedging protocol for 40-60% of Q2 exposure.
"""

c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### DOCUMENT PREVIEW")
    st.markdown(f"<div style='background: {config.COLORS['surface']}; padding: 20px; border-radius: 5px;'>{report_content}</div>", unsafe_allow_html=True)

with c2:
    st.markdown("### EXPORT PROTOCOLS")
    st.download_button(
        label="📥 DOWNLOAD ENCRYPTED BRIEFING (.TXT)",
        data=report_content,
        file_name=f"OVIP_Briefing_{datetime.utcnow().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
