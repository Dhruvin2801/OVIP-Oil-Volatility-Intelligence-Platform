import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Free Hugging Face Model (Mistral 7B)
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
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
    """Finds relevant data and sends it to Hugging Face AI"""
    # 1. Retrieve the 3 most relevant months of data
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    context_text = "\n".join([df.iloc[i]['rag_context'] for i in top_indices])
    
    # 2. Build the Prompt
    prompt = f"<s>[INST] You are OVIP, an oil volatility AI. Answer based ONLY on this data:\n{context_text}\nQuestion: {user_query}[/INST]"
    
    # 3. Securely pull the API Key from Streamlit Secrets
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    payload = {
        "inputs": prompt, 
        "parameters": {"max_new_tokens": 150, "temperature": 0.2, "return_full_text": False}, 
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload).json()
        
        # Handle Hugging Face "Model is loading" sleep errors
        if isinstance(response, dict) and "error" in response:
            return f"⚠️ Hugging Face API is waking up: {response['error']}. Please try asking again in 20 seconds!"
            
        return response[0]['generated_text'].strip()
    except Exception as e:
        return f"⚠️ API Connection Error: {str(e)}"
