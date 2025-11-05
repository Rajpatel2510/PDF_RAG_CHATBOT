import requests
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


class GeminiLLM:
    def __init__(self, model_name="gemini-2.5-flash"):
        API_KEY = GEMINI_API_KEY
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"  
        self.model_name = model_name

    def generate_answer(self, question: str, context: list) -> str:
        try:
            context_text = "\n".join(context) if context else "No document context"
            prompt = (
                f"You are an advanced document analysis assistant capable of providing insightful, detailed, and professional answers.\n"
                f"Your task is to analyze the provided document content and respond accurately to the user's question.\n"
                f"Document Type: General (resumes, reports, legal contracts, manuals, papers, etc.)\n"
                f"Content:\n{context_text}\n\n"
                f"User Question: {question}\n"
                "Answer:"
            )
            response = requests.post(self.url, json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            })

            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                raise RuntimeError(f"Gemini API error: {response.text}")

        except Exception as e:
            return f"Failed to get response: {str(e)}"