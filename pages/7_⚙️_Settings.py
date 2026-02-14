import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="OVIP - Settings", layout="wide")
config.apply_custom_theme()

st.markdown("<h2>⚙️ SYSTEM CONFIGURATION</h2><hr style='border: 1px solid #1E3A5F;'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### SECURITY OVERRIDES")
    hf_override = st.text_input("Hugging Face API Token Override:", type="password", help="Overrides the .streamlit/secrets.toml file if provided.")
    
    if st.button("VERIFY SECRETS"):
        if hf_override:
            st.session_state['hf_override'] = hf_override
            st.success("API Token stored in local session.")
        elif 'HF_TOKEN' in st.secrets:
            st.success("Default API Token located in secure vault.")
        else:
            st.error("No API Token found. AI modules will be restricted.")

with c2:
    st.markdown("### SYSTEM PARAMETERS")
    st.slider("Data Refresh Rate (Seconds):", min_value=10, max_value=300, value=60)
    st.selectbox("Default Geographic Projection:", ["Orthographic (3D)", "Mercator (2D)", "Natural Earth"])
    st.checkbox("Enable Experimental Features", value=False)
    
    st.markdown("""
    <br>
    <div style='border: 1px solid #FF073A; padding: 10px; border-radius: 5px;'>
        <strong style='color: #FF073A;'>DANGER ZONE</strong><br>
        <button style='margin-top: 10px; width: 100%;'>PURGE LOCAL CACHE</button>
    </div>
    """, unsafe_allow_html=True)
