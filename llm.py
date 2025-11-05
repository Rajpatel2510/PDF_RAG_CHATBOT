import requests
import os
from dotenv import load_dotenv
import streamlit as st

if os.getenv("STREAMLIT_RUN_ON_SAVE") is None:
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]

class GeminiLLM:
    def __init__(self, model_name="gemini-2.5-flash"):
        API_KEY = GEMINI_API_KEY
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"  
        self.model_name = model_name

    def generate_answer(self, question: str, context: list = None) -> str:
        try:
            context_text = "\n".join(context) if context else ""

            prompt = f"You are a professional document-grounded AI.\n"

            if context:
                prompt += (
                    f"Use ONLY the document information below.\n"
                    f"If answer is not in document say: \"Not in document\".\n\n"
                    f"Document Context:\n{context_text}\n\n"
                    f"User Question: {question}\nAnswer:"
                )
            else:
                # Self-RAG scoring prompt
                prompt += question

            response = requests.post(self.url, json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            })

            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                raise RuntimeError(f"Gemini API error: {response.text}")

        except Exception as e:
            return f"Failed to get response: {str(e)}"
