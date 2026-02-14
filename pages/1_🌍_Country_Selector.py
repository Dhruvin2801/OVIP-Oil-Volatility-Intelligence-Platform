import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Connect to root configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="OVIP - Target Selection", layout="wide", initial_sidebar_state="expanded")
config.apply_custom_theme()

st.markdown("<h1 style='text-align: center;'>GLOBAL ASSET TRACKER</h1><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

df_globe = pd.DataFrame({
    'Market': ['WTI (USA)', 'Brent (UK)', 'Dubai (UAE)', 'Arab Light (SAU)', 'Urals (RUS)'],
    'Lat': [37.09, 55.37, 23.42, 23.88, 61.52],
    'Lon': [-95.71, -3.43, 53.84, 45.07, 105.31],
    'Risk_Index': [18, 16, 21, 15, 30]
})

c_left, c_mid, c_right = st.columns([1, 2, 1])

with c_mid:
    fig = px.scatter_geo(
        df_globe, lat='Lat', lon='Lon', hover_name='Market', size='Risk_Index',
        projection='orthographic',
        color_discrete_sequence=[config.COLORS['accent_primary']]
    )
    fig.update_layout(
        height=450, margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showland=True, landcolor=config.COLORS['surface'],
            showocean=True, oceancolor=config.COLORS['background'],
            showcountries=True, countrycolor=config.COLORS['border']
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_market = st.selectbox("ENGAGE TARGET MARKET:", df_globe['Market'].tolist())
    
    if st.button("🚀 INITIALIZE COMMAND CENTER", use_container_width=True):
        st.session_state['market_display'] = selected_market
        st.switch_page("pages/2_📊_Dashboard.py")
