
## 🧠 Task 2: Text Chunking, Embedding, and Vector Store Indexing

This task prepares the cleaned consumer complaint narratives for **semantic search and retrieval-augmented generation (RAG)** by:

1. **Chunking** long narratives into manageable segments.
2. **Embedding** each chunk using a sentence-level transformer model.
3. **Indexing** the embeddings into a vector database (ChromaDB) for fast similarity search.

---

### 📂 Input & Output

* **Input:** `../data/filtered_complaints.csv` — Cleaned complaints from Task 1.
* **Output:** Persisted vector index in `../vector_store/chroma_index/`.

---

### 🛠️ Components

#### ✅ Chunking Strategy

* Implemented with `LangChain`'s `RecursiveCharacterTextSplitter`
* **Chunk Size:** 500 characters
* **Chunk Overlap:** 50 characters

> This balances information retention and avoids breaking meaning across chunks, which is crucial for high-quality embeddings.

#### ✅ Embedding Model

* Model: `sentence-transformers/all-MiniLM-L6-v2`
* Justification:

  * Fast and lightweight
  * Strong performance on semantic similarity tasks
  * Widely adopted and supported in `sentence-transformers` and LangChain

#### ✅ Vector Store

* Chosen backend: **ChromaDB**
* Reasons for choosing:

  * Native support in LangChain
  * Easy to use locally and scalable for future deployment
  * Supports storing rich metadata for each document

Each chunk is stored with:

* Original `complaint_id`
* `product` category
* `chunk_id`
* Original chunk `text`

---

### 🚀 How to Run

Make sure you’ve installed dependencies:

```bash
pip install -r requirements.txt
```

Then run the script:

```bash
python src/chunking_embedding.py
```

You will see logs for:

* Loaded complaints
* Number of generated chunks
* Embedding progress
* Indexing progress
* Final persisted vector store path

---

### 📁 Output Directory

The output vector store is saved to:

```
vector_store/chroma_index/
├── chroma-collections.parquet
├── chroma-embeddings.parquet
├── chroma.sqlite3
...
```

> ⚠️ This folder is **excluded from Git** via `.gitignore` to avoid pushing large binary files.

---

### 📝 Notes

* This task sets up the foundation for **Task 3: Retrieval + LLM Q\&A**.
* The indexed vector store enables similarity-based retrieval during inference.

