from langchain_text_splitters import RecursiveCharacterTextSplitter


def text_chunker(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
    chunks = text_splitter.split_documents(documents)
    return chunks