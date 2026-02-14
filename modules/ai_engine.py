import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🚀 UPGRADED: The Universal Chat Completions Router
API_URL = "https://router.huggingface.co/v1/chat/completions"

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
    """Finds relevant data and securely communicates with the AI"""
    # 1. Retrieve the 3 most relevant months of data
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    context_text = "\n".join([df.iloc[i]['rag_context'] for i in top_indices])
    
    # 2. Secure Headers
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    # 3. Modern Payload Format (Using a highly-available, fast model)
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct", 
        "messages": [
            {"role": "system", "content": "You are OVIP_DAEMON, an advanced cybersecurity and oil volatility AI terminal. Keep responses short, hacker-like, professional, and base them strictly on the provided data."},
            {"role": "user", "content": f"CONTEXT_DATA:\n{context_text}\n\nUSER_COMMAND: {user_query}"}
        ],
        "max_tokens": 150,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # 🛡️ FATAL ERROR TRAP: Catch HTML Error Pages before they crash Python!
        if not response.ok:
            return f"⚠️ SYSTEM_ERROR_CODE [{response.status_code}]: Connection refused by provider. Msg: {response.text[:150]}"
            
        data = response.json()
        
        # Parse the OpenAI-style response format
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ UNEXPECTED_PAYLOAD: {str(data)[:150]}"
            
    except Exception as e:
        return f"⚠️ KERNEL_PANIC (System Fault): {str(e)}"
