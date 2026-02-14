import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def setup_rag_vector_db(df):
    """Converts CSV data into a searchable text database"""
    df['rag_context'] = df.apply(lambda x: (
        f"Date: {x['Date'].strftime('%Y-%m-%d')}. "
        f"WTI Price: ${x['WTI']:.2f}. Volatility: {x['Volatility']:.3f}. "
        f"Crisis Probability: {x['Crisis_Prob']:.2f}."
    ), axis=1)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['rag_context'].fillna(""))
    return vectorizer, tfidf_matrix, df

def get_ai_response(user_query, vectorizer, tfidf_matrix, df):
    """Securely communicates with Google Gemini, dynamically selecting the best available model."""
    try:
        # 1. Initialize Gemini API
        genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
        
        # 2. Force the absolute latest date into the prompt
        df_sorted = df.sort_values('Date')
        latest = df_sorted.iloc[-1]
        latest_context = f"CURRENT LIVE DATA (Absolute Latest Date in Database: {latest['Date'].strftime('%Y-%m-%d')}): WTI=${latest['WTI']:.2f}, Volatility={latest['Volatility']:.3f}, Crisis Prob={latest['Crisis_Prob']:.2f}."
        
        # 3. Retrieve Historical Context
        query_vec = vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-3:][::-1]
        hist_context = "\n".join([df.iloc[i]['rag_context'] for i in top_indices])
        
        full_context = f"{latest_context}\n\nHISTORICAL DB MATCHES:\n{hist_context}"

        # 4. Extract Chat Memory
        history_text = ""
        if 'chat' in st.session_state:
            for msg in st.session_state.chat[-6:-1]: 
                sender = "USER" if msg['role'] == 'user' else "OVIP_DAEMON"
                history_text += f"{sender}: {msg['content']}\n"

        # 5. 🛡️ THE BULLETPROOF FIX: Auto-Detect Available Models
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return "⚠️ KERNEL_PANIC: API Key is valid, but Google has not enabled text models for this region/account."
            
        # Target the smartest, fastest flash model available on your key, fallback to whatever is first
        target_model = valid_models[0] 
        for m in valid_models:
            if "2.5-flash" in m:
                target_model = m
                break
            elif "1.5-flash" in m:
                target_model = m

        model = genai.GenerativeModel(target_model)
        
        # 6. Build the Executive Analyst Persona Prompt
        prompt = f"""
        [SYSTEM: INSTRUCTION OVERRIDE]
        You are OVIP, an advanced quantitative intelligence AI designed for oil market operations and risk management.
        Your tone should be professional, highly analytical, and clear—similar to a top-tier financial analyst or a Bloomberg Terminal AI.
        
        CRITICAL DIRECTIVES:
        1. If the user provides market data, analyze it using clear headers like: [EXECUTIVE SUMMARY], [MARKET IMPLICATIONS], and [OPERATIONAL STRATEGY].
        2. If the user asks a basic business question, or asks you to explain a concept (like "What is NPRS?"), drop the heavy jargon. Explain it simply and clearly so a university professor or a corporate executive can understand the business value.
        3. For your internal knowledge: "NPRS-1" stands for "News and Pricing Regime Signal V1". It is our custom Random Forest machine learning model that predicts the directional movement of volatility with 69% accuracy.
        4. Do NOT just repeat the user's input. Synthesize it and provide actionable operational advice.
        
        [LIVE SYSTEM DATA]
        {full_context}
        
        [CHAT MEMORY LOGS]
        {history_text}
        
        USER_COMMAND: {user_query}
        """
