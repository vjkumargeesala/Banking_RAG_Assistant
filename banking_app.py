import os
import math
import re
import tempfile
import streamlit as st
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="🏦 Banking AI Assistant", page_icon="🏦", layout="wide")
st.title("🏦 National Bank AI Assistant")
st.caption("Powered by LLaMA-3.1 via Groq | RAG + Agentic AI | Open Source Stack")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        help="Get free key at console.groq.com — starts with gsk_"
    )

    st.divider()
    st.markdown("### 📄 Upload Banking Documents")
    uploaded = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("📥 Ingest Documents", use_container_width=True):
        if not uploaded:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Processing documents..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                all_chunks = []
                with tempfile.TemporaryDirectory() as d:
                    for f in uploaded:
                        p = Path(d) / f.name
                        p.write_bytes(f.read())
                        try:
                            if f.name.endswith(".pdf"):
                                docs = PyPDFLoader(str(p)).load()
                            else:
                                docs = TextLoader(str(p), encoding="utf-8").load()
                            all_chunks.extend(splitter.split_documents(docs))
                        except Exception as e:
                            st.warning(f"Could not load {f.name}: {e}")

                if all_chunks:
                    st.session_state.vs = FAISS.from_documents(all_chunks, embeddings)
                    st.session_state.chunk_count = len(all_chunks)

            st.success(f"✅ Indexed {st.session_state.get('chunk_count', 0)} chunks!")

    st.divider()
    if "chunk_count" in st.session_state:
        st.metric("Chunks Indexed", st.session_state.chunk_count)
        st.success("Knowledge base ready ✅")
    else:
        st.info("Upload documents to get started")

# ─────────────────────────────────────────────
# SAMPLE QUESTIONS
# ─────────────────────────────────────────────
st.markdown("### 💡 Try asking:")
cols = st.columns(2)
sample_questions = [
    "Am I eligible for a personal loan with ₹30k salary and 720 CIBIL?",
    "Calculate EMI for ₹10 lakh at 12% for 3 years",
    "What documents are needed for a home loan?",
    "What credit card suits an ₹8 lakh annual income?",
]
for i, q in enumerate(sample_questions):
    if cols[i % 2].button(q, use_container_width=True):
        st.session_state.prefill = q

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
question = st.chat_input("Ask about loans, EMI, credit cards, policies...") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not groq_api_key:
        answer = "⚠️ Please enter your Groq API key in the sidebar to use the assistant."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        # ─────────────────────────────────────────────
        # TOOL 1: document_search — matches @tool in notebook
        # ─────────────────────────────────────────────
        @tool
        def document_search(query: str) -> str:
            """Search the bank's internal documents for information about loan policies,
            eligibility criteria, interest rates, required documents, credit card policies,
            customer FAQ, savings account rules, FD rates, and any banking procedures.
            Use this tool FIRST for any question about the bank's products or services.
            Input: a natural-language question or keywords about banking."""
            if "vs" not in st.session_state:
                return "No documents uploaded yet. Please upload banking documents in the sidebar."
            results = st.session_state.vs.similarity_search(query, k=6)
            return "\n\n".join(
                f"[Source {i+1}] {r.page_content}"
                for i, r in enumerate(results)
            )

        # ─────────────────────────────────────────────
        # TOOL 2: loan_emi_calculator — matches @tool in notebook
        # ─────────────────────────────────────────────
        @tool
        def loan_emi_calculator(expression: str) -> str:
            """Calculate loan EMI (Equated Monthly Installment) or evaluate math.
            For EMI: use format 'EMI: principal=500000, rate=12, tenure=36'
              principal = loan amount in rupees
              rate      = annual interest rate in percent (e.g. 12 for 12%)
              tenure    = loan duration in months (e.g. 36 for 3 years)
            For general math: provide a Python expression e.g. '500000 * 0.01'
            Use this tool whenever the user asks for EMI, monthly payment, or loan math."""

            emi_pattern = r"(?i)emi.*principal[=:]\s*([\d.]+).*rate[=:]\s*([\d.]+).*tenure[=:]\s*([\d.]+)"
            emi_match = re.search(emi_pattern, expression)

            if emi_match:
                principal     = float(emi_match.group(1))
                annual_rate   = float(emi_match.group(2))
                tenure_months = float(emi_match.group(3))
                r = annual_rate / 12 / 100
                n = tenure_months
                emi = principal * r * (1 + r)**n / ((1 + r)**n - 1) if r else principal / n
                total_payment  = emi * n
                total_interest = total_payment - principal
                return (
                    f"💰 EMI Calculation:\n"
                    f"   Loan Amount    : ₹{principal:,.0f}\n"
                    f"   Interest Rate  : {annual_rate}% per annum\n"
                    f"   Tenure         : {tenure_months:.0f} months ({tenure_months/12:.1f} years)\n"
                    f"   Monthly EMI    : ₹{emi:,.0f}\n"
                    f"   Total Payment  : ₹{total_payment:,.0f}\n"
                    f"   Total Interest : ₹{total_interest:,.0f}"
                )

            _SAFE = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            _SAFE.update({"abs": abs, "round": round})
            identifiers = re.findall(r"[a-zA-Z_]\w*", expression)
            unsafe = [i for i in identifiers if i not in _SAFE]
            if unsafe:
                return f"Error: unsafe identifiers detected: {unsafe}."
            try:
                result = eval(expression, {"__builtins__": {}}, _SAFE)
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {e}"

        # ─────────────────────────────────────────────
        # TOOL 3: web_search — matches @tool in notebook
        # ─────────────────────────────────────────────
        @tool
        def web_search(query: str) -> str:
            """Search the web using DuckDuckGo for current banking information such as
            RBI repo rate, current market interest rates, recent RBI policy updates,
            or any general information not found in the bank's internal documents.
            Input: a search query string."""
            return DuckDuckGoSearchRun().run(query)

        # ─────────────────────────────────────────────
        # AGENT — matches notebook Phase 5 exactly
        # ─────────────────────────────────────────────
        tools = [document_search, loan_emi_calculator, web_search]

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=512,
        )

        react_prompt = hub.pull("hwchase17/react")

        agent_exec = AgentExecutor(
            agent=create_react_agent(llm=llm, tools=tools, prompt=react_prompt),
            tools=tools,
            handle_parsing_errors=True,
            max_iterations=6,
            verbose=False,
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = agent_exec.invoke({"input": question})["output"]
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
