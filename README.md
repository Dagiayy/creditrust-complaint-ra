
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

3. **Retrieval + LLM Response (Task 3):**

   * Retrieve relevant complaint chunks using semantic similarity.
   * Generate user-friendly answers with a lightweight local LLM.
   * Post-process to reduce hallucination and repetition.

---

## 📁 Project Structure

```
creditrust-complaint-ra/
├── data/                          
│   ├── complaints.csv             
│   ├── filtered_complaints.csv    
│   └── filtered_complaints_expanded.csv  
│
├── notebooks/                     
│   └── eda_preprocessing.ipynb    
│
├── src/                           
│   ├── preprocessing.py           
│   ├── chunking_embedding.py      
│   └── rag_pipeline.py            # Task 3: Retrieval + LLM-based answering
│
├── vector_store/                  
│   └── chroma_index/              
│
├── .gitignore                     
├── README.md                      
├── requirements.txt               
└── report.md                      
```

---

## 🚀 How to Run

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Task 1: Data Cleaning & EDA

```bash
jupyter notebook notebooks/eda_preprocessing.ipynb
```

### 3. Task 2: Chunking + Embedding + Indexing

```bash
python src/chunking_embedding.py
```

### 4. Task 3: Retrieval + LLM Answering

```bash
python src/rag_pipeline.py
```

You’ll be prompted to enter a natural language question like:

```
What are common issues with student loans?
```

The system will retrieve relevant complaint excerpts and generate an answer.

---

## 📚 Tasks Completed

### ✅ Task 1: Data Preprocessing

* Filtered dataset to key financial products (e.g., Credit Card, BNPL).
* Optionally expanded filtering to include other categories.
* Cleaned text (lowercasing, punctuation removal, empty entries).
* Visualized product distributions.

### ✅ Task 2: Chunking & Indexing

* Split long narratives into overlapping text chunks.
* Embedded using `sentence-transformers/all-MiniLM-L6-v2`.
* Persisted vector index with ChromaDB + metadata.

### ✅ Task 3: Retrieval + Generation

* Retrieved top-k relevant complaint chunks using semantic similarity.
* Used **Falcon-RW-1B** (2.6GB) for local question answering.
* Applied structured prompts to reduce hallucination.
* Added post-processing to clean repetitive answers.

---

## ⚙️ Lightweight LLM Model Used

* **Model:** `tiiuae/falcon-rw-1b`
* **Why:** Fast, CPU-friendly, <3GB download, good for offline/local inference
* **Alternative:** Replace with any `AutoModelForCausalLM`-compatible model (e.g. Mistral, TinyLlama)

---

## 🧪 Sample Questions to Try

```text
What are common issues with student loans?
What complaints relate to identity theft?
Are there issues related to paying off debt but still being contacted?
What are the common reasons for loan application rejection?
```

---

## 📦 Dependencies

* `pandas`, `seaborn`, `matplotlib`
* `langchain`, `sentence-transformers`, `chromadb`
* `transformers`, `torch`, `scikit-learn`
* `tqdm`, `pickle`

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📌 Why RAG for Complaints?

Financial institutions face thousands of customer complaints. RAG enables:

* Faster, context-aware search
* Insight extraction from noisy narrative data
* Enhanced customer service with natural language Q\&A

---

## 💬 Contact

Built for internal use at **Kifiya**.

