
# 📊 Credit Complaint RAG System

An end-to-end pipeline for transforming millions of consumer financial complaints into a **retrieval-augmented generation (RAG)** system using **semantic search** and **LLM-based question answering**.

---

## 📌 Overview

This project enables intelligent Q\&A over financial complaint narratives filed by consumers. It follows these key stages:

1. **Data Cleaning & Filtering (Task 1):**

   * Preprocess millions of complaints.
   * Focus on key financial products.
   * Explore narrative availability & product distribution.

2. **Text Chunking & Vector Indexing (Task 2):**

   * Break long narratives into coherent text chunks.
   * Embed using a sentence-level model.
   * Index chunks in a vector store (ChromaDB) with rich metadata.

3. **(Upcoming) Retrieval + LLM Response (Task 3):**

   * Retrieve relevant complaint chunks using semantic similarity.
   * Generate user-friendly answers with a Large Language Model.

---

## 📁 Project Structure

```
creditrust-complaint-ra/
├── data/                          # Input and output CSV files
│   ├── complaints.csv             # Raw complaint dataset
│   ├── filtered_complaints.csv    # Cleaned dataset (strict product filter)
│   └── filtered_complaints_expanded.csv  # Optional expanded filter output
│
├── notebooks/                     # EDA and preprocessing analysis
│   └── eda_preprocessing.ipynb    # Task 1: Exploratory notebook
│
├── src/                           # Core processing scripts
│   ├── preprocessing.py           # Data cleaning and product filtering
│   └── chunking_embedding.py      # Chunking + embedding + vector store
│
├── vector_store/                  # Vector database (excluded from Git)
│   └── chroma_index/              # ChromaDB persistent index
│
├── .gitignore                     # Exclude large or unnecessary files
├── README.md                      # Project overview (this file)
├── requirements.txt               # Dependencies
└── report.md                      # Documentation for each task
```

---

## 🚀 How to Run

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Task 1: Data Cleaning & EDA

```bash
# Open notebook
jupyter notebook notebooks/eda_preprocessing.ipynb
```

### 3. Task 2: Chunking + Embedding + Indexing

```bash
python src/chunking_embedding.py
```

Outputs:

* `vector_store/chroma_index/`: vector index and metadata

---

## 📚 Tasks Completed

### ✅ Task 1: Data Preprocessing

* Filtered dataset to key financial products (e.g., Credit Card, BNPL).
* Optionally expanded filtering to include other valuable categories.
* Cleaned text narratives (lowercasing, punctuation, empty removals).
* Visualized product distribution and narrative availability.

### ✅ Task 2: Chunking & Indexing

* Split long narratives into overlapping chunks.
* Embedded using `all-MiniLM-L6-v2`.
* Indexed with ChromaDB, including complaint metadata.

---

## 🤖 Next Steps

* **Task 3:** Implement RAG pipeline:

  * Retrieve top-k similar complaint chunks.
  * Generate a response using an LLM (e.g., OpenAI, Mistral, etc.).
* Add evaluation methods (e.g., retrieval quality, hallucination rate).

---

## 📦 Dependencies

* `pandas`, `seaborn`, `matplotlib`
* `langchain`, `sentence-transformers`
* `chromadb`, `faiss-cpu` (optional if switching back)
* `scikit-learn`, `tqdm`, `pickle`

Install everything:

```bash
pip install -r requirements.txt
```

---

## 📌 Why RAG for Complaints?

Financial institutions face thousands of customer complaints. RAG enables:

* Faster, context-aware search
* Insight extraction from noisy narrative data
* Enhanced customer service with natural language interfaces

---

## 💬 Contact

Built for internal use at **Kifiya**
