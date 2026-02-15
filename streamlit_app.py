# ==========================================
# 3. DIRECTORY-AWARE DATA UPLINK
# ==========================================
@st.cache_data
def load_data():
    # Adding the 'data/' folder path so the system knows where to look
    files = ['data/merged_final.csv', 'data/merged_final_corrected.csv', 'merged_final.csv']
    
    for f in files:
        if os.path.exists(f): 
            try:
                df = pd.read_csv(f)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                print(f"Error loading {f}: {e}")
                pass
    
    # EMERGENCY PROXY DATA
    st.warning("⚠️ PROXY_DATA: Database offline. Generating tactical simulation.")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=150)
    wti = np.linspace(70, 85, 150) + np.random.normal(0, 2, 150)
    vol = np.linspace(0.1, 0.25, 150) + np.random.normal(0, 0.02, 150)
    return pd.DataFrame({'Date': dates, 'WTI': wti, 'Volatility': vol, 'Crisis_Prob': np.zeros(150)})

df_main = load_data()
