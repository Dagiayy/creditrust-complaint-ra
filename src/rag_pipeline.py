import os
import torch  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore

# Constants
CHROMA_DIR = "../vector_store/chroma_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "tiiuae/falcon-rw-1b"
TOP_K = 5
MAX_NEW_TOKENS = 256

# Load embedding model and Chroma vector store
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Load Falcon RW-1B causal language model
print("🔁 Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.7,
    device=torch.device("cpu")
)

# Refined prompt template
PROMPT_TMPL = """You are a helpful assistant for a financial complaints company.
Use only the following complaint excerpts to answer the question.

Summarize the common problems mentioned by customers. Do not make up information.
Avoid repeating sentences or listing generic facts.

Context:
{context}

Question:
{question}

Answer (in 3-4 sentences):
"""


def ask_question(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PROMPT_TMPL.format(context=context, question=question)
    response = llm_pipe(prompt)[0]["generated_text"].strip()
    # Trim the prompt from the generated output (Falcon may repeat input)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response, docs

def main():
    print("✅ RAG pipeline ready.")
    while True:
        q = input("\nEnter your question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            answer, docs = ask_question(q)
            print("\n---\n👩‍💼 Answer:\n", answer)
            print("\n📚 Sources:")
            for doc in docs:
                print("—", doc.metadata.get("complaint_id"), doc.metadata.get("product"), ":", doc.page_content[:200], "...")
        except Exception as e:
            print("❌ Error:", e)

if __name__ == "__main__":
    main()
