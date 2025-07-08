
# 📊 Credit Complaint RAG System

An end-to-end pipeline for transforming millions of consumer financial complaints into a **Retrieval-Augmented Generation (RAG)** system using **semantic search**, **vector databases**, and **LLM-based question answering**, accessible via an interactive web interface.

---

## 📌 Overview

This project enables intelligent Q\&A over financial complaint narratives filed by consumers. It follows these key stages:

1. **Task 1 — Data Cleaning & Filtering**

   * Preprocess and clean raw complaint data.
   * Focus on key financial products (e.g., Credit Cards, BNPL).
   * Explore narrative availability & visualize distribution.

2. **Task 2 — Text Chunking & Vector Indexing**

   * Split narratives into meaningful chunks.
   * Embed chunks with sentence-level model.
   * Store in ChromaDB vector database.

3. **Task 3 — Retrieval + LLM Response Generation**

   * Use semantic search to retrieve relevant complaints.
   * Generate answers using a lightweight LLM (Falcon-RW-1B).
   * Post-process output to reduce hallucination/repetition.

4. **Task 4 — Interactive Chat Interface**

   * Build a web UI with **Streamlit** for non-technical users.
   * Display both the answer and the supporting complaint excerpts.
   * Provide clear, minimal, and professional user experience.

---

## 📁 Project Structure

```
creditrust-complaint-ra/
├── app.py                        # Task 4: Streamlit chat interface
├── data/                         # Raw and filtered datasets
│   ├── complaints.csv
│   ├── filtered_complaints.csv
│   └── filtered_complaints_expanded.csv
│
├── notebooks/
│   └── eda_preprocessing.ipynb  # Task 1: Data cleaning & EDA
│
├── screenshots/                 # Screenshots of the final web interface
│
├── src/
│   ├── preprocessing.py         # Task 1 script
│   ├── chunking_embedding.py    # Task 2 script
│   └── rag_pipeline.py          # Task 3: CLI-based QA pipeline
│
├── vector_store/                # ChromaDB persistent index (excluded in Git)
│   └── chroma_index/
│
├── .gitignore
├── README.md                    # This file
├── report.md                    # Final task report
└── requirements.txt             # Project dependencies
```

---

## 🚀 How to Run the Project

### 1. ✅ Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ Ensure you have Python 3.8+ and enough memory (\~3GB RAM minimum for LLM loading).

---

### 2. 📊 Task 1: Data Cleaning & EDA

```bash
jupyter notebook notebooks/eda_preprocessing.ipynb
```

---

### 3. 📐 Task 2: Chunking + Embedding + Indexing

```bash
python src/chunking_embedding.py
```

> This creates the Chroma vector store (`vector_store/chroma_index/`).

---

### 4. 🧠 Task 3: Retrieval + Answer Generation (CLI Mode)

```bash
python src/rag_pipeline.py
```

Then type a question like:

```text
What are common issues with student loans?
```

The CLI will:

* Retrieve relevant complaint chunks
* Generate an answer using the LLM
* Display answer and metadata sources

---

### 5. 💬 Task 4: Web Chat Interface (Recommended)

```bash
streamlit run app.py
```

This launches a modern web interface where users can:

* Enter questions via a textbox
* View generated answers
* See the actual complaint chunks (sources) used for response
* Reset/clear the session

💡 The UI features a **light dark background** for readability and a professional look.

---

## 🧪 Sample Questions to Try

```text
What are common issues with student loans?
Why are customers upset even after paying off debt?
What complaints relate to identity theft?
What are the common reasons for loan application rejection?
```

---

## 🤖 Model Used

| Attribute    | Value                                                |
| ------------ | ---------------------------------------------------- |
| Model        | `tiiuae/falcon-rw-1b` (2.6GB)                        |
| Framework    | HuggingFace Transformers                             |
| Why?         | Lightweight, CPU-compatible, offline-ready           |
| Alternatives | Mistral 7B (if GPU available), TinyLlama, GPTQ, etc. |

---

## 🧠 How It Works

1. **Vector Search:** Uses `all-MiniLM-L6-v2` to convert complaint chunks to dense embeddings.
2. **ChromaDB:** Stores embeddings and retrieves top-k matches.
3. **LLM Prompting:** Combines relevant chunks into a prompt for LLM.
4. **Streaming Answer (in Web UI):** Shows token-by-token output for better UX.
5. **Source Display:** Always shows which complaint excerpts were used for transparency.

---

## 📦 Dependencies

```text
pandas
numpy
matplotlib
seaborn
streamlit
langchain
sentence-transformers
chromadb
transformers
torch
scikit-learn
tqdm
pickle
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📸 Screenshots

All screenshots of the working chatbot interface are saved in the `screenshots/` folder. You can include them in your report or presentation.

---

## 📌 Why RAG for Complaints?

Financial service providers face thousands of complex, narrative-heavy complaints. This system:

* Enables natural language search across complaint data.
* Helps internal staff, analysts, and QA teams extract insights fast.
* Builds transparency by showing source excerpts with generated answers.
* Works offline and respects data privacy.

---

## 💬 Contact

Built for internal use at **Kifiya Financial Technologies**
Project Lead: `Dagmawi Ayenew`
Mentor: `Kifiya AI Research Team`

