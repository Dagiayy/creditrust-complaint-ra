"""Streamlit chat interface for the CrediTrust Complaint RAG system.

Thin presentation layer only — all retrieval/generation logic lives in
`creditrust.rag.pipeline.RAGPipeline`, shared with the FastAPI service and CLI.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from creditrust.config import get_settings  # noqa: E402
from creditrust.logging_config import get_logger  # noqa: E402
from creditrust.rag.pipeline import RAGPipeline  # noqa: E402
from creditrust.rag.retriever import vector_store_exists  # noqa: E402

logger = get_logger(__name__)
settings = get_settings()

st.set_page_config(
    page_title="CrediTrust AI Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .big-title { font-size: 32px; font-weight: 800; margin-bottom: 8px; }
        .sub-title { font-size: 17px; color: #888; margin-bottom: 25px; }
        .answer-box {
            background-color: #262626; padding: 1.3em; border-radius: 10px;
            border: 1px solid #444; font-size: 16px; line-height: 1.6; margin-top: 10px;
        }
        .source-box {
            background-color: #262626; padding: 1em; margin-top: 12px;
            border-radius: 8px; border-left: 5px solid #3498db; font-size: 14px;
        }
        .warn-box {
            background-color: #4a3b1a; padding: 1em; border-radius: 8px;
            border-left: 5px solid #e0a72e; font-size: 14px; margin-top: 10px;
        }
        .footer { font-size: 13px; color: #aaa; margin-top: 3em; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading RAG pipeline (embedding model + LLM)...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings(settings)


st.markdown('<div class="big-title">💬 CrediTrust AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about real financial complaints — '
    "AI will retrieve and summarize the answers.</div>",
    unsafe_allow_html=True,
)

if not vector_store_exists(settings.vector_store_dir):
    st.error(
        "⚠️ No vector index found. Run `python scripts/run_pipeline.py all` "
        "to preprocess the data and build the index before using this app."
    )
    st.stop()

with st.form("qa_form"):
    question = st.text_input(
        "🔍 Enter your question:",
        placeholder="e.g., Why do customers complain after paying off debts?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("🧠 Thinking..."):
        try:
            pipeline = load_pipeline()
            result = pipeline.ask(question)

            st.markdown("### ✅ AI Answer")
            box_class = "answer-box" if result.had_sufficient_context else "warn-box"
            st.markdown(
                f'<div class="{box_class}">{html.escape(result.answer)}</div>',
                unsafe_allow_html=True,
            )

            if result.sources:
                st.markdown("### 📚 Supporting Complaints")
                for source in result.sources:
                    preview = source.excerpt.replace("\n", " ")
                    preview = preview[:400] + ("..." if len(preview) > 400 else "")
                    st.markdown(
                        f'<div class="source-box"><b>Complaint ID:</b> {html.escape(source.complaint_id)} '
                        f"&nbsp;&nbsp; <b>Product:</b> {html.escape(source.product)}<br>"
                        f"{html.escape(preview)}</div>",
                        unsafe_allow_html=True,
                    )
            st.caption(f"Answered in {result.latency_seconds:.2f}s")
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
            logger.exception("Failed to answer question: %r", question)
            st.error(f"❌ Error: {exc}")

st.markdown(
    '<div class="footer">Built for CrediTrust · Financial Complaint RAG System</div>',
    unsafe_allow_html=True,
)
