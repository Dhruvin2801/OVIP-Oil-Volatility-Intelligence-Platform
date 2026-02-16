import streamlit as st
import sys
from pathlib import Path

# Add root directory to path so imports work perfectly
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import config
from modules.data_loader import get_data_loader
from modules.ai_engine import setup_rag_vector_db, get_ai_response

# 1. Page Configuration
st.set_page_config(page_title="OVIP - Intelligence Terminal", layout="wide")
config.apply_custom_theme()

# 2. Header UI
st.markdown("<h2>💬 INTELLIGENCE TERMINAL</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

# 3. Optimized Data & Vector DB Loading
@st.cache_resource
def initialize_rag():
    """Caches the heavy vector DB setup so it doesn't reload on every click"""
    df = get_data_loader().merge_all_data()
    vec, tfidf, rag_df = setup_rag_vector_db(df)
    return vec, tfidf, rag_df

vec, tfidf, rag_df = initialize_rag()

# 4. Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "SECURE UPLINK ESTABLISHED. Query the market data."}
    ]

# 5. Display History
for msg in st.session_state.chat_history:
    bg_color = config.COLORS['surface'] if msg['role'] == 'user' else "rgba(100, 255, 218, 0.05)"
    border_color = "#8892B0" if msg['role'] == 'user' else config.COLORS['accent_primary']
    st.markdown(f"""
    <div style='background: {bg_color}; padding: 15px; border-radius: 5px; border-left: 3px solid {border_color}; margin-bottom: 10px;'>
        <strong style='color:{config.COLORS['accent_primary']};'>{'YOU' if msg['role'] == 'user' else 'OVIP AI'}:</strong><br>
        <span style='color: #CCD6F6;'>{msg['content']}</span>
    </div>
    """, unsafe_allow_html=True)

# 6. Combined Chat Logic (The Fix)
if prompt := st.chat_input("Enter command..."):
    # Step A: Append User Message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Step B: Generate AI Response in the SAME run
    with st.spinner("Processing Intelligence Report..."):
        try:
            ans = get_ai_response(prompt, vec, tfidf, rag_df)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Engine Error: {str(e)}")
    
    # Step C: Rerun ONCE to show the final state
    st.rerun()
