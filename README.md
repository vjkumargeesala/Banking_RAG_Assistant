# 🏦 Banking Agentic RAG Assistant

> An AI-powered banking assistant that answers loan eligibility questions, calculates EMIs, responds to customer FAQs, and fetches live RBI updates — built on an open-source LLM stack with no paid APIs except Groq's free tier.

---

## 🚀 Live Demo

```bash
streamlit run banking_app.py
```

---

## 🏗️ Architecture

```
                        USER QUESTION
                              │
                              ▼
              ┌───────────────────────────────┐
              │         ReAct Agent           │
              │   LLaMA-3.1-8B via Groq       │
              │  "Which tool should I use?"   │
              └──────┬──────────┬─────────────┘
                     │          │            │
                     ▼          ▼            ▼
            ┌─────────────┐ ┌──────────┐ ┌──────────────┐
            │ RAG Tool    │ │Calculator│ │  Web Search  │
            │ FAISS +     │ │  EMI /   │ │ DuckDuckGo   │
            │ MiniLM-L6   │ │  Math    │ │  Live RBI    │
            └──────┬──────┘ └────┬─────┘ └──────┬───────┘
                   └─────────────┴───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   LLaMA-3.1-8B-Instant │
                    │  Generates final answer │
                    │  using retrieved context│
                    └────────────────────────┘
```

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| LLM | LLaMA-3.1-8B-Instant via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Agent Framework | LangChain ReAct (langchain-classic) |
| Web Search | DuckDuckGo (no API key needed) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## ✨ Features

- **ReAct Agent** — Reasons step-by-step and picks the right tool for each query
- **RAG Pipeline** — Retrieves from 110+ indexed chunks across 3 banking documents
- **EMI Calculator** — Custom tool for accurate loan math (no LLM hallucination)
- **Live Web Search** — Fetches current RBI rates via DuckDuckGo
- **Streamlit Chat UI** — Clean chat interface with sample question buttons
- **Document Upload** — Upload your own PDF/TXT banking documents at runtime

---

## 📊 Evaluation Results

| Metric | Value |
|---|---|
| Average Keyword Hit Rate | **95.8%** |
| Average Retrieval Latency | **<0.01s** |
| Chunks Indexed | **110+** |
| Test Cases | **6 domain-specific queries** |
| Perfect Scoring Questions | **5 / 6** |

### Detailed Retrieval Results

| Category | Keywords Hit | Top Source | Latency |
|---|---|---|---|
| Loan Eligibility | 4/4 (100%) | customer_faq.txt | 0.010s |
| Credit Card | 4/4 (100%) | customer_faq.txt | 0.009s |
| Customer FAQ | 4/4 (100%) | customer_faq.txt | 0.010s |
| Loan Documents | 4/4 (100%) | loan_policy.txt | 0.009s |
| Credit Card Fees | 4/4 (100%) | credit_card_policy.txt | 0.008s |
| Fixed Deposit | 3/4 (75%) | customer_faq.txt | 0.007s |

---

## 📁 Project Structure

```
banking-rag-assistant/
├── 📓 Banking_RAG_Assistant.ipynb   ← Main notebook (all phases)
├── 🖥️  banking_app.py               ← Streamlit web application
├── 📄 requirements.txt              ← Python dependencies
├── 📄 .gitignore                    ← Ignores API keys & index files
├── 📂 banking_docs/
│   ├── loan_policy.txt              ← Personal & home loan policies
│   ├── customer_faq.txt             ← Customer FAQ document
│   └── credit_card_policy.txt       ← Credit card terms & features
└── 📂 faiss_index/                  ← Auto-generated (gitignored)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/banking-rag-assistant.git
cd banking-rag-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up → API Keys → Create Key
- Key starts with `gsk_`

### 4. Run the Streamlit app
```bash
streamlit run banking_app.py
```

### 5. Upload documents & start chatting
- Paste your Groq API key in the sidebar
- Upload the files from `banking_docs/` folder
- Click **Ingest Documents**
- Start asking questions!

---

## 💬 Example Queries

| Type | Example |
|---|---|
| Loan Eligibility | "I earn ₹35,000/month with CIBIL 720. Am I eligible for a personal loan?" |
| EMI Calculation | "What is my EMI for ₹30 lakh home loan at 9% for 20 years?" |
| Credit Card | "What credit card suits someone with ₹8 lakh annual income?" |
| Document Query | "What documents do I need for a home loan?" |
| Live Web Search | "What is the current RBI repo rate?" |
| Multi-step | "I want ₹8 lakh personal loan — what rate will I get and what's my EMI for 4 years?" |

---

## 🔧 Running the Notebook

Open `Banking_RAG_Assistant.ipynb` and run cells phase by phase:

| Phase | Description |
|---|---|
| 0 | Install dependencies |
| 1 | Configuration & settings |
| 2 | Create banking documents |
| 3 | Build RAG pipeline + FAISS index |
| 4 | Define agent tools |
| 5 | Initialize LLaMA-3.1 via Groq + ReAct agent |
| 6 | Run example queries |
| 7 | Interactive chat loop |
| 8 | Retrieval evaluation |
| 9 | Save & reload FAISS index |
| 10 | Launch Streamlit app |

---

## 🔑 Environment Variables

Never hardcode API keys. Use a `.env` file:

```bash
GROQ_API_KEY=gsk_your_key_here
```

```python
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

---

## 📦 Requirements

```
langchain
langchain-community
langchain-classic
langchain-huggingface
langchain-groq
langchain-text-splitters
langchainhub
faiss-cpu
pypdf
sentence-transformers
duckduckgo-search
streamlit
python-dotenv
pandas
```

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first.

---

## 📝 License

MIT License — free to use and modify.

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end agentic AI development with open-source LLMs.
