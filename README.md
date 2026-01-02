# Cybertrace

CyberTrace is a **Retrieval-Augmented Generation (RAG) based Cyber Threat Intelligence chatbot** that enables time-aware and explainable querying over cybersecurity documents using FAISS, FastAPI, and Gemini.

---

## 📁 Folder Structure (VERY IMPORTANT)

Ensure your project directory looks **exactly** like this:
project/
│
├── member_one_rag_full.py
├── .env
├── requirements.txt
│
├── faiss/
│ └── pdf_store/
│ ├── index.faiss
│ ├── id2text.pkl
│ ├── id2meta.pkl
│
└── (optional) check.py

👉 The script expects `faiss/pdf_store/` relative to where you run it.

---

## 🧪 Create & Activate Virtual Environment (Recommended)

### Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

🔐 Environment Setup (CRITICAL)

Create a .env file in the same directory as member_one_rag_full.py.

GOOGLE_API_KEY=your_gemini_api_key_here

🧠 Test FAISS Store (Optional but Recommended)

Run:

python check.py

▶️ Run the FastAPI Server
uvicorn member_one_rag_full:app --reload

🌐 Quick Browser Test

Open:

http://127.0.0.1:8000/


Expected response:

{
  "message": "🚀 CyberTrace RAG Backend (PDF + Time-Aware Retrieval) is running"
}

🔍 Query the RAG System
Option 1: Swagger UI

Open:

http://127.0.0.1:8000/docs


Example request:

{
  "query": "What ransomware was used by FIN8?",
  "top_k": 3
}

Option 2: curl
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d "{\"query\": \"What ransomware was used by FIN8?\", \"top_k\": 3}"
