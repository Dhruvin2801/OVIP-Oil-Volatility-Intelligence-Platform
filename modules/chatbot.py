import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Free Hugging Face Model (Mistral 7B)
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

class OVIPAssistant:
    def __init__(self, df):
        self.vectorizer, self.tfidf_matrix, self.rag_df = self._setup_knowledge_base(df)

    def _setup_knowledge_base(self, df):
        """Translates CSV numerical data into narrative text for the LLM."""
        df = df.copy()
        
        # Build sentences that the AI can actually read
        df['rag_context'] = df.apply(lambda x: (
            f"On {x['Date'].strftime('%Y-%m-%d')}: "
            f"WTI Crude Price was ${x.get('WTI', 0):.2f}. "
            f"Market Volatility was {x.get('Volatility', 0):.3f}. "
            f"Crisis Probability was {x.get('Crisis_Prob', 0):.2f}. "
            f"News Sentiment Score was {x.get('Score', 0):.2f}."
        ), axis=1)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['rag_context'].fillna(""))
        return vectorizer, tfidf_matrix, df

    def get_response(self, user_query):
        """Retrieves relevant history and queries the LLM."""
        # 1. RAG Search (Find Top 3 most relevant days)
        query_vec = self.vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-3:][::-1]
        context_text = "\n".join([self.rag_df.iloc[i]['rag_context'] for i in top_indices])
        
        # 2. Strict Prompt Construction
        prompt = f"""<s>[INST] You are OVIP, an elite AI for oil market risk analysis.
Answer the user's question using ONLY the provided market data below.
If you don't know the answer based on the data, say "I don't have enough data to answer that."

MARKET DATA:
{context_text}

USER QUESTION:
{user_query}
[/INST]"""

        # 3. Call Hugging Face API
        try:
            hf_token = st.secrets.get('HF_TOKEN', None)
            if not hf_token:
                return "⚠️ API Error: HF_TOKEN not found in .streamlit/secrets.toml"

            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {
                "inputs": prompt, 
                "parameters": {"max_new_tokens": 150, "temperature": 0.2, "return_full_text": False},
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"Chatbot Error: {str(e)}")
            return f"⚠️ Secure Connection Failed. The model may be waking up, please try again in 10s. (Error: {str(e)})"
