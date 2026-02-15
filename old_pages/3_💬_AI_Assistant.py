import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.data_loader import get_data_loader
from modules.ai_engine import setup_rag_vector_db, get_ai_response

st.set_page_config(page_title="OVIP - Intelligence Terminal", layout="wide")
config.apply_custom_theme()

st.markdown("<h2>💬 INTELLIGENCE TERMINAL</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

df = get_data_loader().merge_all_data()
vec, tfidf, rag_df = setup_rag_vector_db(df)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "SECURE UPLINK ESTABLISHED. Query the market data."}]

for msg in st.session_state.chat_history:
    bg_color = config.COLORS['surface'] if msg['role'] == 'user' else "rgba(100, 255, 218, 0.05)"
    border_color = "#8892B0" if msg['role'] == 'user' else config.COLORS['accent_primary']
    st.markdown(f"""
    <div style='background: {bg_color}; padding: 15px; border-radius: 5px; border-left: 3px solid {border_color}; margin-bottom: 10px;'>
        <strong>{'YOU' if msg['role'] == 'user' else 'OVIP AI'}:</strong> {msg['content']}
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input("Enter command..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("Processing..."):
        ans = get_ai_response(st.session_state.chat_history[-1]["content"], vec, tfidf, rag_df)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()
