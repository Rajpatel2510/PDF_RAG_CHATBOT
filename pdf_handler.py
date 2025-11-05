# this file extract text from pdf
from langchain_community.document_loaders import PyMuPDFLoader
import re


def extract_text(file_path : str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    cleaned_docs = []
    for doc in documents:
        text = doc.page_content
        text = re.sub(r'\s+', ' ', text)              # remove extra spaces/newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # remove non-ASCII
        text = re.sub(r'Page\s*\d+', '', text)        # remove page numbers

        doc.page_content = text.strip()
        cleaned_docs.append(doc)

    return cleaned_docs