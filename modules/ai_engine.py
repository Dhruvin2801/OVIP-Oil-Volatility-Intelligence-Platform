import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def setup_rag_vector_db(df):
    """Optimized: Shortens context strings to save tokens and prevent lag."""
    df['rag_context'] = df.apply(lambda x: (
        f"{x['Date'].strftime('%m/%y')}: WTI=${x['WTI']:.1f}, Vol={x['Volatility']:.2f}, CP={x['Crisis_Prob']:.2f}"
    ), axis=1)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['rag_context'].fillna(""))
    return vectorizer, tfidf_matrix, df

def get_ai_response(user_query, vectorizer, tfidf_matrix, df):
    """Switches brain to Groq Llama-3 for instant, unlimited responses."""
    try:
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        
        # 1. Grab absolute latest data
        latest = df.sort_values('Date').iloc[-1]
        latest_context = f"CURRENT_STATE: {latest['Date'].strftime('%Y-%m-%d')} | WTI: ${latest['WTI']:.2f} | Vol: {latest['Volatility']:.3f} | CP: {latest['Crisis_Prob']:.2f}"
        
        # 2. Get history memory
        history_text = ""
        if 'chat' in st.session_state:
            for msg in st.session_state.chat[-5:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # 3. Call Groq (Using Llama-3-70b for high intelligence)
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are OVIP, an elite oil market analyst. Use professional headers: [EXECUTIVE SUMMARY], [MARKET IMPLICATIONS], [STRATEGY]. Explain technical terms like 'NPRS-1' simply if asked."},
                {"role": "user", "content": f"DATA: {latest_context}\n\nHISTORY:\n{history_text}\n\nCOMMAND: {user_query}"}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT: {str(e)}"
