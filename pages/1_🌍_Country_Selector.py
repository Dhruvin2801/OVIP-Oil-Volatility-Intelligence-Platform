import streamlit as st
import plotly.express as px
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

st.set_page_config(page_title="OVIP // INIT_SEQUENCE", layout="wide", initial_sidebar_state="collapsed")
config.apply_custom_theme()

st.markdown("<h1 style='text-align: center;'>root@ovip:~# ./initialize_target.sh</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #008F11 !important;'>[WAITING FOR USER INPUT]</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px dashed #00FF41;'>", unsafe_allow_html=True)

c_left, c_mid, c_right = st.columns([1, 2, 1])

with c_mid:
    # Cybersecurity Wireframe Globe
    df_globe = pd.DataFrame({
        'Target': ['TARGET: WTI (USA)', 'TARGET: BRENT (UK)', 'TARGET: DUBAI (UAE)'], 
        'Lat': [37.09, 55.37, 23.42], 
        'Lon': [-95.71, -3.43, 53.84], 
        'Threat_Level': [100, 80, 85]
    })
    
    fig = px.scatter_geo(
        df_globe, lat='Lat', lon='Lon', hover_name='Target', size='Threat_Level', 
        projection='orthographic', color_discrete_sequence=['#FF003C']
    )
    
    fig.update_geos(
        bgcolor='rgba(0,0,0,0)',
        showland=False, showcoastlines=True, coastlinecolor='#00FF41',
        showcountries=True, countrycolor='#008F11',
        showframe=True, framecolor='#00FF41',
        showocean=False,
        projection_rotation={'lon': -40, 'lat': 20, 'roll': 0}
    )
    
    fig.update_layout(
        height=450, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("```bash\n> SELECT_TARGET_MARKET_NODE\n```")
    market = st.selectbox("", ["🇺🇸 TARGET_NODE_01: WTI_CRUDE", "🇬🇧 TARGET_NODE_02: BRENT_CRUDE", "🇦🇪 TARGET_NODE_03: DUBAI_CRUDE"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("EXECUTE // BYPASS MAINFRAME"):
        # Fix: Save to session state so the Dashboard can read it
        st.session_state['market_display'] = market
        st.switch_page("pages/2_📊_Dashboard.py")
