import chromadb
from typing import List
from embedder import Embedder  # your sentence-transformer (or similar)

class ChromaDBHandler:
    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "rag_collection"):
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            self.embedder = Embedder()
        except Exception as e:
            raise RuntimeError(f"ChromaDB initialization failed: {str(e)}")

    def add_data(self, file_name: str, documents: List[str], embeddings: List[List[float]]):
        try:
            if not documents:
                raise ValueError("No documents to add.")
            if len(documents) != len(embeddings):
                raise ValueError("documents and embeddings length mismatch.")

            ids = [f"{file_name}_{i}" for i in range(len(documents))]
            metadatas = [{"source": file_name} for _ in documents]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return f"Added {len(documents)} chunks for {file_name}"
        except Exception as e:
            raise RuntimeError(f"Error adding data to ChromaDB: {str(e)}")
        
    def search_similar_chunks(self, file_name: str, query: str, top_k: int = 5):
        query_emb = self.embedder.encode([query])

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"source": file_name}
        )

        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            return []

        return list(zip(docs, dists))

    def persist(self):
        try:
            self.client.persist()
        except Exception as e:
            raise RuntimeError(f"Failed to persist ChromaDB: {str(e)}")
