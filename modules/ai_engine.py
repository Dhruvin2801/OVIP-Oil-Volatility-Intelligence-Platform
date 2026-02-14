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
        
        # 6. Build the Persona Prompt
        prompt = f"""
        You are OVIP_DAEMON, an advanced cybersecurity and oil volatility AI terminal. 
        Keep responses short, hacker-like, professional, and base them strictly on the provided data. 
        You have access to the recent chat history to answer follow-up questions like 'why?'
        
        [SYSTEM INJECTION: CONTEXT DATA]
        {full_context}
        
        [SYSTEM INJECTION: RECENT CHAT MEMORY]
        {history_text}
        
        USER_COMMAND: {user_query}
        """
        
        # 7. Generate Response
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"⚠️ KERNEL_PANIC (Gemini API Fault): {str(e)}"
