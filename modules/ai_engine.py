import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def setup_rag_vector_db(df):
    """Bulletproof RAG setup with no lambdas to prevent TypeErrors."""
    working_df = df.copy()
    
    # Ensure columns exist before processing
    cols = ['WTI', 'Volatility', 'Crisis_Prob', 'Date']
    for col in cols:
        if col not in working_df.columns:
            if col == 'Date':
                working_df[col] = pd.Timestamp.now()
            else:
                working_df[col] = 0.0

    # Build context strings using a safe list comprehension
    contexts = []
    for _, row in working_df.iterrows():
        try:
            d = row['Date'].strftime('%m/%y') if hasattr(row['Date'], 'strftime') else "N/A"
            contexts.append(f"{d}: WTI=${row['WTI']:.1f}, Vol={row['Volatility']:.2f}, CP={row['Crisis_Prob']:.2f}")
        except:
            contexts.append("SYSTEM_NODE_OFFLINE")
            
    working_df['rag_context'] = contexts
    
    # Vector Search Setup
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(working_df['rag_context'])
    
    return vectorizer, tfidf_matrix, working_df

def get_ai_response(query, vec, tfidf, df):
    """Tactical Groq Query Engine."""
    try:
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        latest = df.iloc[-1]
        
        system_stats = f"WTI: ${latest['WTI']:.2f} | VOL: {latest['Volatility']:.3f} | CP: {latest['Crisis_Prob']:.2f}"
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are OVIP_DAEMON. Elite tactical analyst. Tone: Cold, professional, hacker-like. Use [EXECUTIVE SUMMARY]."},
                {"role": "user", "content": f"SYSTEM_STATS: {system_stats}\n\nQUERY: {query}"}
            ],
            temperature=0.1,
            max_tokens=600
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ UPLINK_ERROR: {str(e)}"
