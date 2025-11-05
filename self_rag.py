class SelfRAG:
    def __init__(self, llm):
        self.llm = llm  # GeminiLLM object

    def evaluate_context(self, question, chunks):
        context_text = "\n\n---\n".join(chunks)

        judge_prompt = f"""
                You are a Self-RAG evaluator.
                Your job: Check if retrieved text is relevant to the question.

                Question:
                {question}

                Retrieved Chunks:
                {context_text}

                Score relevance 0-1. 
                Return only this JSON format:

                {{
                "avg_score": <score>,
                "decision": "good" or "bad"
                }}
                """

        return self.llm.generate_answer(judge_prompt)

    def final_answer(self, question, chunks):
        context_text = "\n\n---\n".join(chunks)

        answer_prompt = f"""
                Answer ONLY using this context.
                If the answer is not in the document, say:
                "Not in document."

                Context:
                {context_text}

                Question:
                {question}
                """
        return self.llm.generate_answer(answer_prompt)
