import streamlit as st
import plotly.graph_objects as go
import time
import sys, os
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from modules.data_loader import get_data_loader
from modules.ai_engine import setup_rag_vector_db, get_ai_response

st.set_page_config(page_title="OVIP // COMMAND_CENTER", layout="wide", initial_sidebar_state="collapsed")
config.apply_custom_theme()

# 1. Simulated Hack Boot Sequence (Only runs once)
if 'booted' not in st.session_state:
    boot_box = st.empty()
    for i in range(1, 4):
        boot_box.markdown(f"```bash\n[SYSTEM] Decrypting database... {i*33}%\n[SYSTEM] Bypassing firewalls...\n[SYSTEM] Establishing secure tunnel...```")
        time.sleep(0.3)
    boot_box.empty()
    st.session_state['booted'] = True

# 2. Data Loading
loader = get_data_loader()
df_main = loader.merge_all_data()
metrics = loader.get_latest_metrics()

if df_main.empty or metrics is None:
    st.error(">>> FATAL_ERROR: /data/ payload missing. Connection terminated.")
    st.stop()

# 3. Header & Live Time
market = st.session_state.get('market_display', '🇺🇸 TARGET_NODE_01: WTI_CRUDE')
st.markdown(f"<h2>root@ovip:~# monitor {market.split(':')[0]}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #008F11;'>SESSION_ACTIVE // TIMESTAMP: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px dashed #00FF41;'>", unsafe_allow_html=True)

# 4. Top Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("WTI_PRICE_INDEX", f"${metrics['price']:.2f}", f"{metrics['price_change']:+.2f}%")
c2.metric("VOLATILITY_SIGMA", f"{metrics['volatility']:.3f}", "STABLE")

regime_color = "#FF003C" if "CRISIS" in metrics['regime'] else "#00FF41"
c3.markdown(f"""
<div style='background-color: #050505; border: 1px solid {regime_color}; border-left: 5px solid {regime_color}; padding: 15px; box-shadow: 0 0 10px {regime_color};'>
    <span style='font-size:10px; color:#008F11;'>[REGIME_STATE]</span><br>
    <h3 style='color:{regime_color} !important; margin:0;'>{metrics['regime']}</h3>
    <span style='color:{regime_color};'>Prob: {metrics['crisis_prob']:.2f}</span>
</div>
""", unsafe_allow_html=True)

c4.metric("NPRS-1_SIGNAL_OVERRIDE", "▲ UP (HEDGE)", "CONF_INTERVAL: 69.1%")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Main Grid
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("### > HISTORICAL_VOLATILITY_RADAR")
    # Neon Plotly Chart
    fig = go.Figure()
    
    # Grid lines to look like radar
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#003300'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#003300')
    )
    
    fig.add_trace(go.Scatter(
        x=df_main['Date'], y=df_main['Volatility'], name='Vol',
        line=dict(color='#00FF41', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 65, 0.05)'
    ))
    
    # Blood-red Crisis Dots
    crisis_df = df_main[df_main['Crisis_Prob'] > 0.5]
    fig.add_trace(go.Scatter(
        x=crisis_df['Date'], y=crisis_df['Volatility'], mode='markers', name='Crisis Node',
        marker=dict(color='#FF003C', size=8, symbol='cross', line=dict(color='#FF003C', width=2))
    ))
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("### > THREAT_MATRIX")
    st.markdown("""
    ```bash
    GEOPOLITICAL_RISK [||||||||--] 80%  [WARN]
    SUPPLY_SHOCK      [|||-------] 30%  [SAFE]
    NLP_SENTIMENT     [|||||||||-] 95%  [CRIT]
    ```
    """)
    
    st.markdown("### > SECURE_AI_TERMINAL")
    vec, tfidf, rag_df = setup_rag_vector_db(df_main)
    
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "OVIP_DAEMON ONLINE. AWAITING QUERY..."}]
        
    for msg in st.session_state.chat:
        user_color = "#008F11" if msg['role'] == 'user' else "#00FF41"
        sender = "root@user" if msg['role'] == 'user' else "system@ovip"
        st.markdown(f"<p style='color: {user_color}; margin: 0;'><b>{sender}:~$</b> {msg['content']}</p>", unsafe_allow_html=True)

    if prompt := st.chat_input("> EXECUTE COMMAND..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.rerun()
        
    if st.session_state.chat[-1]["role"] == "user":
        with st.spinner("PROCESSING_QUERY..."):
            ans = get_ai_response(st.session_state.chat[-1]["content"], vec, tfidf, rag_df)
            st.session_state.chat.append({"role": "assistant", "content": ans})
            st.rerun()
