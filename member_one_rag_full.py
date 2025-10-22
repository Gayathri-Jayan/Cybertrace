import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from typing import List, Optional, Dict, Any
import faiss
import pickle
import numpy as np

# --------------------------
# LLM Configuration
# --------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("RAG_MODEL_NAME", "text-bison-001")

HAS_GOOGLE = False
HAS_OPENAI = False
genai = None
openai = None

if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        HAS_GOOGLE = True
        print("✅ Google Generative AI configured")
    except Exception:
        print("⚠️ Failed to configure Google Generative AI")
        traceback.print_exc()

if not HAS_GOOGLE and OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        HAS_OPENAI = True
        print("✅ OpenAI configured")
    except Exception:
        print("⚠️ Failed to configure OpenAI")
        traceback.print_exc()

if not HAS_GOOGLE and not HAS_OPENAI:
    print("⚠️ No LLM provider configured. Using dummy fallback.")

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="RAG Pipeline API")

# --------------------------
# Embedding Models (multiple)
# --------------------------
EMBEDDING_MODELS = {
    "pdf_store": "all-mpnet-base-v2",
    "csv_store": "all-mpnet-base-v2",
    "json_store": "all-mpnet-base-v2",
    "url_store": "all-MiniLM-L6-v2"
}

_embedding_models: Dict[str, Any] = {}

def get_embedding_model(store_name: str):
    if store_name not in _embedding_models:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = EMBEDDING_MODELS.get(store_name)
            _embedding_models[store_name] = SentenceTransformer(model_name)
            print(f"✅ Loaded embedding model for {store_name}: {model_name}")
        except Exception:
            print(f"⚠️ Failed to load embedding model for {store_name}")
            traceback.print_exc()
            raise
    return _embedding_models[store_name]

def embed_query(query: str, store_name: str):
    model = get_embedding_model(store_name)
    emb = model.encode([query], normalize_embeddings=True)
    return np.array(emb[0], dtype="float32")

# --------------------------
# FAISS Index Loader
# --------------------------
FAISS_BASE_DIR = "faiss"
INDEX_NAMES = ["pdf_store", "csv_store", "json_store", "url_store"]
faiss_indexes: Dict[str, Dict[str, Any]] = {}

def load_faiss_index(index_name: str):
    try:
        index_dir = os.path.join(FAISS_BASE_DIR, index_name)
        faiss_file = os.path.join(index_dir, "index.faiss")
        possible_meta = [
            os.path.join(index_dir, "index.pkl"),
            os.path.join(index_dir, "metadata.pkl"),
        ]
        if not os.path.exists(faiss_file):
            print(f"⚠️ Skipping {index_name}: missing index file")
            return None, None

        index = faiss.read_index(faiss_file)

        meta = None
        for p in possible_meta:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        meta = pickle.load(f)
                except ModuleNotFoundError:
                    print(f"⚠️ Module missing when loading {p}. Skipping metadata.")
                break

        print(f"✅ Loaded {index_name}: ntotal={index.ntotal}, dim={index.d}")
        return index, meta
    except Exception:
        print(f"⚠️ Failed to load {index_name}")
        traceback.print_exc()
        return None, None

for name in INDEX_NAMES:
    idx, meta = load_faiss_index(name)
    if idx:
        faiss_indexes[name] = {"index": idx, "metadata": meta}

# --------------------------
# FAISS Search
# --------------------------
def search_faiss(query: str, store_name: str, k: int = 5):
    results = []
    if store_name not in faiss_indexes:
        return results

    store = faiss_indexes[store_name]
    index = store.get("index")
    metadata = store.get("metadata")

    q_emb = embed_query(query, store_name)
    if q_emb.shape[0] != index.d:
        print(f"⚠️ Dimension mismatch for {store_name}: query={q_emb.shape[0]}, index={index.d}")
        return results

    try:
        D, I = index.search(q_emb.reshape(1, -1), k)
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            entry_meta = metadata[idx] if metadata else None
            results.append({
                "store": store_name,
                "doc_id": int(idx),
                "metadata": entry_meta,
                "distance": float(dist)
            })
    except Exception:
        print(f"⚠️ Error searching {store_name}")
        traceback.print_exc()

    return results

def retrieve_all(query: str, top_k: int = 5):
    all_results = []
    for store_name in faiss_indexes.keys():
        res = search_faiss(query, store_name, k=top_k)
        all_results.extend(res)
    all_results.sort(key=lambda x: x.get("distance", float("inf")))
    return all_results[:top_k]

def assemble_context(retrieved: List[dict], max_chars: int = 4000) -> str:
    pieces = []
    total = 0
    for r in retrieved:
        text = r.get("metadata")
        if isinstance(text, (dict, list)):
            text = str(text)
        if not text:
            continue
        if total + len(text) > max_chars:
            pieces.append(text[:max_chars - total])
            break
        pieces.append(text)
        total += len(text)
    return "\n---\n".join(pieces)

# --------------------------
# Generate Answer
# --------------------------
def generate_answer(query: str, retrieved: List[dict], provider: Optional[str] = None) -> Dict[str, Any]:
    context = assemble_context(retrieved)
    prompt = f"Use the context to answer the question.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    if provider == "google" or (provider is None and HAS_GOOGLE):
        try:
            resp = genai.generate_text(model=MODEL_NAME, prompt=prompt)
            text = getattr(resp, "text", str(resp))
            return {"answer": text, "model": f"google:{MODEL_NAME}", "raw": resp}
        except Exception:
            traceback.print_exc()

    if provider == "openai" or (provider is None and HAS_OPENAI):
        try:
            completion = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            text = completion["choices"][0]["message"]["content"]
            return {"answer": text, "model": f"openai:{os.getenv('OPENAI_CHAT_MODEL','gpt-3.5-turbo')}", "raw": completion}
        except Exception:
            traceback.print_exc()

    # Fallback
    summary = " \n ".join([str(r.get("metadata")) for r in retrieved[:3]])
    fallback = f"(No LLM configured) Retrieved context summary:\n{summary}\n\nQuestion: {query}\nAnswer: I don't have an LLM configured."
    return {"answer": fallback, "model": "none", "raw": None}

# --------------------------
# FastAPI Endpoints
# --------------------------
@app.get("/")
async def root():
    return {"message": "RAG Pipeline API is running"}

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        top_k = int(data.get("top_k", 5))
        provider = data.get("provider")

        retrieved = retrieve_all(query, top_k)
        out = generate_answer(query, retrieved, provider=provider)

        return {
            "query": query,
            "answer": out.get("answer"),
            "model": out.get("model"),
            "retrieved": retrieved
        }

    except HTTPException as e:
        raise e
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
