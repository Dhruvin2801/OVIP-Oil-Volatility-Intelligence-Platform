import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def setup_rag_vector_db(df):
    """Safely builds the RAG context with zero chance of TypeError."""
    
    # 1. Create a deep copy to avoid modifying the original dataframe
    working_df = df.copy()

    # 2. Force essential columns to exist with default values if missing
    if 'WTI' not in working_df.columns: working_df['WTI'] = 0.0
    if 'Volatility' not in working_df.columns: working_df['Volatility'] = 0.0
    
    # Check for Crisis_Prob or fallback to Regime_Prob
    if 'Crisis_Prob' not in working_df.columns:
        working_df['Crisis_Prob'] = working_df['Regime_Prob'] if 'Regime_Prob' in working_df.columns else 0.0

    # 3. Generate the text context for the AI
    def build_string(row):
        try:
            date_str = row['Date'].strftime('%m/%y') if hasattr(row['Date'], 'strftime') else "N/A"
            return f"{date_str}: WTI=${row['WTI']:.1f}, Vol={row['Volatility']:.2f}, CP={row['Crisis_Prob']:.2f}"
        except:
            return "Data point unavailable"

    working_df['rag_context'] = working_df.apply(build_string, axis=1)
    
    # 4. Initialize Vector Search
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(working_df['rag_context'].fillna(""))
    
    return vectorizer, tfidf_matrix, working_df

def get_ai_response(query, vec, tfidf, df):
    """Secure tactical query via Groq."""
    try:
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        latest = df.iloc[-1]
        
        # Build a safe context for the prompt
        cp_val = latest.get('Crisis_Prob', 0.0)
        wti_val = latest.get('WTI', 0.0)
        vol_val = latest.get('Volatility', 0.0)
        
        context = f"LIVE_SYSTEM_DATA: WTI=${wti_val:.2f}, Vol={vol_val:.3f}, Crisis_Prob={cp_val:.2f}"
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are OVIP_DAEMON, an elite tactical oil analyst. Use tactical headers like [EXECUTIVE SUMMARY]."},
                {"role": "user", "content": f"{context}\n\nUSER_COMMAND: {query}"}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT (AI_ENGINE): {str(e)}"
