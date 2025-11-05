from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            # self.model = SentenceTransformer(model_name)
            self.model = SentenceTransformer(model_name, device="cpu")
            # print(f"Model loaded: {model_name}")

        except Exception as error:
            print("Error loading model. Check model name or internet.")
            raise error

    def encode(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts)

            return embeddings.tolist()

        except Exception as error:
            print("Error while generating embeddings.")
            raise error
