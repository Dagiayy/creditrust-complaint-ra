# Task 1: Exploratory Data Analysis and Data Preprocessing

## 📌 Objective

This task focuses on understanding and preparing the CFPB consumer complaint dataset for integration into a Retrieval-Augmented Generation (RAG) system. It blends **domain-specific analysis** (focused on business relevance) with **applied technical EDA** (focused on data quality and ML-readiness) to ensure the dataset aligns with both stakeholder goals and downstream model requirements.

---

## 🧪 Steps Performed

### 1. Initial Data Loading
- Loaded the full CFPB complaint dataset (`complaints.csv`) containing ~9.6 million records.
- Identified key structural properties including column names, missing values, and basic statistics.

### 2. Domain-Specific EDA
- Inspected and ranked the top 20 most frequent product types.
- Identified top excluded product categories when filtering to a strict product list of 5 business-prioritized products:
  - `Credit card`, `Personal loan`, `Buy Now, Pay Later (BNPL)`, `Savings account`, and `Money transfers`
- Visualized excluded product categories to determine their potential future value in RAG pipelines.

### 3. Applied EDA
- Analyzed distribution of consumer complaint narrative lengths to inform future chunking and embedding strategies.
- Quantified narrative coverage:
  - **With narratives:** ~3M records
  - **Without narratives:** ~6.6M records

### 4. Preprocessing Pipeline
Performed modular preprocessing via `src/preprocessing.py`:
- Filtered to target products (strict or expanded mode)
- Removed rows with empty complaint narratives
- Cleaned narratives (lowercased, removed boilerplate, stripped special characters)
- No word count filtering applied (to preserve all potentially useful narratives)

---

## ✅ Outputs

| File | Description |
|------|-------------|
| `data/filtered_complaints.csv` | Cleaned data with **strict** product filtering (aligned to business scope) |
| `data/filtered_complaints_expanded.csv` | Cleaned data with **expanded** product filtering (for broader RAG tuning) |
| `notebooks/eda_preprocessing.ipynb` | Full EDA and preprocessing workflow (visualizations + processing steps) |
| `src/preprocessing.py` | Modular preprocessing pipeline supporting filter modes and text cleaning |

---

## 🧠 Insights

- The dataset is heavily skewed toward credit reporting and debt collection categories. These are excluded in strict mode but retained in expanded mode for optional future use.
- Around 800K to 1.2M usable narratives are preserved after filtering depending on strict or expanded mode.
- The pipeline is flexible and allows toggling filtering strategies and thresholds for tuning retrieval and generation accuracy in RAG systems.

---

## 🗂 Folder Structure

```

creditrust-complaint-rag/
│
├── data/
│   ├── complaints.csv                 # Original raw dataset (not tracked in Git)
│   ├── filtered\_complaints.csv        # Cleaned with strict filtering
│   └── filtered\_complaints\_expanded.csv # Cleaned with expanded filtering
│
├── notebooks/
│   └── eda\_preprocessing.ipynb        # Interactive analysis + preprocessing
│
├── src/
│   └── preprocessing.py               # Modular data preparation logic
│
├── README.md                          # You are here
└── ...

```

---

## 🔄 Next Steps

- Move to **chunking and embedding** for the cleaned dataset.
- Tune narrative filters and chunking strategies using applied metrics.
- Integrate processed data into RAG indexing and retrieval.

```
