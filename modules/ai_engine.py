import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def setup_rag_vector_db(df):
    def get_val(row, col): return row[col] if col in row else 0.0
    
    # Safely building RAG context
    df['rag_context'] = df.apply(lambda x: (
        f"{x['Date'].strftime('%m/%y') if 'Date' in x else 'N/A'}: "
        f"WTI=${get_val(x, 'WTI'):.1f}, Vol={get_val(x, 'Volatility'):.2f}, "
        f"CP={get_val(x, 'Crisis_Prob', get_val(x, 'Regime_Prob')):.2f}"
    ), axis=1)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['rag_context'].fillna(""))
    return vectorizer, tfidf_matrix, df

def get_ai_response(query, vec, tfidf, df):
    try:
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        latest = df.iloc[-1]
        cp = latest.get('Crisis_Prob', latest.get('Regime_Prob', 0.0))
        context = f"LIVE_DATA: WTI=${latest.get('WTI', 0):.2f}, Vol={latest.get('Volatility', 0):.3f}, CP={cp:.2f}"
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are OVIP_DAEMON, an elite tactical oil analyst. Use tactical headers: [EXECUTIVE SUMMARY], [MARKET IMPLICATIONS]."},
                {"role": "user", "content": f"{context}\n\nCOMMAND: {query}"}
            ],
            temperature=0.2, max_tokens=600
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT: {str(e)}"
