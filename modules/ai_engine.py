import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def setup_rag_vector_db(df):
    df['rag_context'] = df.apply(lambda x: f"{x['Date'].strftime('%m/%y')}: WTI=${x['WTI']:.1f}, Vol={x['Volatility']:.2f}, CP={x['Crisis_Prob']:.2f}", axis=1)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['rag_context'].fillna(""))
    return vectorizer, tfidf_matrix, df

def get_ai_response(user_query, vectorizer, tfidf_matrix, df):
    try:
        client = Groq(api_key=st.secrets['GROQ_API_KEY'])
        latest = df.iloc[-1]
        context = f"LIVE_DATA: WTI=${latest['WTI']:.2f}, Vol={latest['Volatility']:.3f}, CP={latest['Crisis_Prob']:.2f}"
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are OVIP, an elite oil analyst. Use tactical headers: [EXECUTIVE SUMMARY], [MARKET IMPLICATIONS]."},
                {"role": "user", "content": f"{context}\n\nCOMMAND: {user_query}"}
            ],
            temperature=0.2, max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ SYSTEM_FAULT: {str(e)}"
