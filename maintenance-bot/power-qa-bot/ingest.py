"""
文档解析 & 向量入库
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = "data"
FAISS_DIR = "faiss_db"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


def load_documents(data_dir: str):
    docs = []
    for file in Path(data_dir).rglob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue
        print(f"Loading: {file.name}")
        docs.extend(loader.load())
    return docs


def ingest():
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("Please set SILICONFLOW_API_KEY in .env file")

    print("Loading documents...")
    documents = load_documents(DATA_DIR)
    if not documents:
        print("No documents found in data/ directory")
        return

    print(f"Loaded {len(documents)} document segments")

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", ";", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and saving to FAISS...")
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1",
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"Done! Vector DB saved to {FAISS_DIR}/")


if __name__ == "__main__":
    ingest()
