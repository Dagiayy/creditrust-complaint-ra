import streamlit as st
from src.rag_core import ask_question

# --- Page Configuration ---
st.set_page_config(
    page_title="CrediTrust AI Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        body {
           background-color: #d3d3d3;

        }
        .big-title {
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .sub-title {
            font-size: 17px;
            color: #555;
            margin-bottom: 25px;
        }
        .answer-box {
            background-color: #575454;
            padding: 1.3em;
            border-radius: 10px;
            border: 1px solid #ccc;
            font-size: 16px;
            line-height: 1.6;
            margin-top: 10px;
        }
        .source-box {
            background-color: #575454;
            padding: 1em;
            margin-top: 12px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            font-size: 14px;
        }
        .footer {
            font-size: 13px;
            color: #aaa;
            margin-top: 3em;
            text-align: center;
        }
        .stTextInput>div>div>input {
            background-color: #f4f4f4;
            color: #000;
            font-size: 16px;
            border-radius: 6px;
        }
        .stButton>button {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="big-title">💬 CrediTrust AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions about real financial complaints — AI will retrieve and summarize the answers.</div>', unsafe_allow_html=True)

# --- Input Form ---
with st.form("qa_form"):
    question = st.text_input("🔍 Enter your question:", placeholder="e.g., Why do customers complain after paying off debts?")
    submitted = st.form_submit_button("Ask")

# --- Display Answer & Sources ---
if submitted and question.strip():
    with st.spinner("🧠 Thinking..."):
        try:
            answer, docs = ask_question(question)

            # --- Display the AI Answer ---
            st.markdown("### ✅ AI Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # --- Show Supporting Sources ---
            st.markdown("### 📚 Supporting Complaints")
            for doc in docs:
                cid = doc.metadata.get("complaint_id", "Unknown")
                product = doc.metadata.get("product", "Unknown")
                excerpt = doc.page_content.strip().replace("\n", " ")
                preview = excerpt[:400] + ("..." if len(excerpt) > 400 else "")
                st.markdown(
                    f'<div class="source-box"><b>Complaint ID:</b> {cid} &nbsp;&nbsp; <b>Product:</b> {product}<br>{preview}</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# --- Footer ---
st.markdown('<div class="footer">Built with ❤️ by Kifiya · CrediTrust RAG System · 2025</div>', unsafe_allow_html=True)
