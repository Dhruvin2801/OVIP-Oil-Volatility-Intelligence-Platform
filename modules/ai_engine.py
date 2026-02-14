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
    """Securely communicates with Groq using the latest Llama 3.3 model."""
    try:
        # Initialize Groq client with your secret key
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        
        # 1. Grab absolute latest data from the CSV
        df_sorted = df.sort_values('Date')
        latest = df_sorted.iloc[-1]
        latest_context = f"CURRENT_STATE: {latest['Date'].strftime('%Y-%m-%d')} | WTI: ${latest['WTI']:.2f} | Vol: {latest['Volatility']:.3f} | CP: {latest['Crisis_Prob']:.2f}"
        
        # 2. Extract Chat Memory for follow-up questions
        history_text = ""
        if 'chat' in st.session_state:
            for msg in st.session_state.chat[-5:]:
                role = "USER" if msg['role'] == 'user' else "OVIP_DAEMON"
                history_text += f"{role}: {msg['content']}\n"

        # 3. Call Groq using the NEW llama-3.3-70b model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are OVIP, an elite oil market analyst and risk intelligence daemon. "
                        "Tone: Professional, analytical, and hacker-like. "
                        "Use headers: [EXECUTIVE SUMMARY], [MARKET IMPLICATIONS], [STRATEGY]. "
                        "Explain terms like 'NPRS-1' (69% accuracy directional model) if asked."
                    )
                },
                {"role": "user", "content": f"DATA_CONTEXT: {latest_context}\n\nCHAT_HISTORY:\n{history_text}\n\nUSER_COMMAND: {user_query}"}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT: {str(e)}"
