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
    """Securely communicates with Groq using the latest Llama 3.3 model with expanded memory."""
    try:
        # Initialize Groq client
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        
        # 1. Grab absolute latest data from the CSV
        df_sorted = df.sort_values('Date')
        latest = df_sorted.iloc[-1]
        latest_context = f"CURRENT_STATE: {latest['Date'].strftime('%Y-%m-%d')} | WTI: ${latest['WTI']:.2f} | Volatility Sigma: {latest['Volatility']:.3f} | Crisis Prob: {latest['Crisis_Prob']:.2f}"
        
        # 2. Extract Chat Memory
        history_text = ""
        if 'chat' in st.session_state:
            # We take the last 6 messages to maintain deep context
            for msg in st.session_state.chat[-6:]:
                role = "USER" if msg['role'] == 'user' else "OVIP_DAEMON"
                history_text += f"{role}: {msg['content']}\n"

        # 3. Call Groq with high-capacity token limit
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are OVIP, an elite tactical oil analyst and risk intelligence daemon. "
                        "Tone: Professional, cold, analytical, and tactical. "
                        "MANDATE: You must synthesize the hypothetical user queries with the actual LIVE SYSTEM DATA. "
                        "Example: If asked about war, compare the 'what-if' to the current Volatility Sigma and Crisis Prob. "
                        "Always use clear tactical headers: [EXECUTIVE SUMMARY], [MARKET THREAT LEVEL], [OPERATIONAL STRATEGY]. "
                        "Define 'NPRS-1' as the 69% accuracy directional ML model if conceptually relevant."
                    )
                },
                {"role": "user", "content": f"LIVE_DATA: {latest_context}\n\nCHAT_LOGS:\n{history_text}\n\nUSER_COMMAND: {user_query}"}
            ],
            temperature=0.2, # Keeps it very factual/analytical
            max_tokens=1000  # Increased to prevent response cutoff
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT: {str(e)}"
