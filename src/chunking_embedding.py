import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import numpy as np

# Constants
VECTOR_STORE_DIR = "../vector_store/chroma_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 5000

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def chunk_texts(texts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    all_chunks = []
    for idx, text in enumerate(texts):
        chunks = splitter.split_text(str(text))
        chunk_data = [{"chunk": c, "source_index": idx, "chunk_id": i} for i, c in enumerate(chunks)]
        all_chunks.extend(chunk_data)
    return all_chunks

def sanitize_metadata(meta: dict) -> dict:
    # Ensures metadata is in allowed format for Chroma
    return {
        k: (
            str(v) if isinstance(v, (np.int64, np.float64, np.bool_)) else v
        )
        for k, v in meta.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }

def main():
    print("📄 Loading data...")
    df = load_data("data/filtered_complaints.csv")
    print(f"📄 Loaded {len(df)} complaints for chunking.")

    print("✂️ Chunking text...")
    chunks = chunk_texts(df["cleaned_narrative"].tolist())
    print(f"✂️ Created {len(chunks)} text chunks.")

    print("🧠 Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("📦 Initializing Chroma vector store...")
    vectorstore = Chroma(
        embedding_function=embedder,
        persist_directory=VECTOR_STORE_DIR
    )

    # Prepare texts and metadata
    texts = [chunk["chunk"] for chunk in chunks]
    metadatas = [
        sanitize_metadata({
            "complaint_id": str(df.iloc[chunk["source_index"]].get("Complaint ID", chunk["source_index"])),
            "product": df.iloc[chunk["source_index"]]["Product"],
            "chunk_id": chunk["chunk_id"]
        })
        for chunk in chunks
    ]

    print("📥 Inserting into ChromaDB in batches...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        vectorstore.add_texts(batch_texts, metadatas=batch_metas)

    print("💾 Persisting Chroma vector store...")
    vectorstore.persist()

    print(f"✅ Done! Vector store saved to: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    main()
