import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.data_loader import get_data_loader

st.set_page_config(page_title="OVIP - Analytics", layout="wide")
config.apply_custom_theme()

st.markdown("<h2>📈 MULTI-VARIATE ANALYTICS</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

df = get_data_loader().merge_all_data()

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Sentiment vs Volatility")
    if 'Score' in df.columns:
        fig_scatter = px.scatter(
            df, x='Score', y='Volatility', color='Crisis_Prob',
            color_continuous_scale=[config.COLORS['accent_primary'], config.COLORS['danger']],
            template='plotly_dark'
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.markdown("### Feature Correlation")
    cols = [c for c in ['Volatility', 'WTI', 'Score', 'Intensity', 'Crisis_Prob'] if c in df.columns]
    corr = df[cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=[config.COLORS['background'], config.COLORS['accent_primary']],
        template='plotly_dark'
    )
    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_corr, use_container_width=True)
