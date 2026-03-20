"""
local_chatbot.py
Fully local RAG chatbot — zero API key, zero external service.
Stack: ChromaDB + sentence-transformers (embeddings) + Flan-T5 (generation)
"""

import os
import json
import re
import warnings
warnings.filterwarnings("ignore")

BASE        = os.path.dirname(__file__)
KB_PATH     = os.path.join(BASE, "data", "knowledge_base.json")
CHROMA_DIR  = os.path.join(BASE, "chroma_db")
COLLECTION  = "risklens_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-base"

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_collection = None
_generator  = None


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store():
    """Build ChromaDB from knowledge_base.json. Called once at Docker build."""
    import chromadb
    from chromadb.utils import embedding_functions

    print("[VectorDB] Loading knowledge base...")
    with open(KB_PATH, "r") as f:
        documents = json.load(f)
    print(f"[VectorDB] {len(documents)} documents loaded")

    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    ids, texts, metas = [], [], []
    for doc in documents:
        ids.append(doc["id"])
        texts.append(f"{doc['title']}\n\n{doc['content']}")
        metas.append({
            "disease":    doc["disease"],
            "category":   doc["category"],
            "risk_level": doc["risk_level"],
            "title":      doc["title"],
        })

    for i in range(0, len(ids), 10):
        col.add(
            ids=ids[i:i+10],
            documents=texts[i:i+10],
            metadatas=metas[i:i+10]
        )
        print(f"[VectorDB] Indexed {min(i+10, len(ids))}/{len(ids)}")

    print(f"[VectorDB] ✅ Built — {col.count()} documents in store")


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils import embedding_functions
        ef          = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION, embedding_function=ef)
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(query: str, diseases: list, risk_scores: dict, n: int = 4):
    """Semantic search — returns (context_str, [source_titles])"""
    col = _get_collection()

    where = None
    if diseases:
        dl = diseases + ["General"]
        where = {"disease": dl[0]} if len(dl) == 1 else {"disease": {"$in": dl}}

    res = col.query(
        query_texts=[query],
        n_results=min(n, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    if not res["documents"] or not res["documents"][0]:
        return "", []

    parts, titles = [], []
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        sim = 1 - dist
        if sim < 0.25:
            continue
        risk_note = ""
        if risk_scores and meta["disease"] in risk_scores:
            risk_note = f" | User risk: {risk_scores[meta['disease']]}%"
        parts.append(f"[{meta['title']}{risk_note}]\n{doc[:900]}")
        titles.append(meta["title"])

    return "\n\n---\n\n".join(parts), titles


# ─────────────────────────────────────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_generator():
    global _generator
    if _generator is not None:
        return _generator
    try:
        from transformers import pipeline
        print("[LLM] Loading Flan-T5-base...")
        _generator = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_new_tokens=280,
            do_sample=False,
        )
        print("[LLM] ✅ Flan-T5 ready")
    except Exception as e:
        print(f"[LLM] ⚠️  Could not load: {e} — using retrieval-only mode")
        _generator = None
    return _generator


def _generate(question: str, context: str, risk_scores: dict) -> str | None:
    gen = _load_generator()
    if not gen:
        return None

    risk_summary = ", ".join(
        f"{d}: {s}%" for d, s in (risk_scores or {}).items()
    ) or "not provided"

    prompt = (
        f"You are a medical health assistant for RiskLens.\n"
        f"User's disease risk profile: {risk_summary}\n\n"
        f"Use ONLY the medical knowledge below to answer.\n"
        f"Knowledge:\n{context[:1400]}\n\n"
        f"Question: {question}\n"
        f"Answer in 3-4 clear sentences:"
    )

    try:
        out    = gen(prompt, max_new_tokens=260, do_sample=False)
        answer = out[0]["generated_text"].strip()
        if "Answer" in answer:
            answer = answer.split("Answer")[-1].lstrip(":").strip()
        return answer if len(answer) > 15 else None
    except Exception as e:
        print(f"[LLM] Generation error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK — pure retrieval answer
# ─────────────────────────────────────────────────────────────────────────────

def _retrieval_answer(question: str, context: str, risk_scores: dict) -> str:
    if not context:
        return (
            "I don't have specific information on that in my knowledge base. "
            "Please consult a qualified doctor for personalised advice."
        )

    stop_words = {
        "what","how","is","are","can","the","a","an","i","my",
        "do","does","should","will","tell","me","about"
    }
    keywords = set(
        re.sub(r'[^\w\s]', '', question.lower()).split()
    ) - stop_words

    scored = []
    for line in context.split("\n"):
        line = line.strip()
        if len(line) < 50:
            continue
        score = sum(1 for k in keywords if k in line.lower())
        if score:
            scored.append((score, line))

    scored.sort(key=lambda x: -x[0])
    best = [s[1] for s in scored[:3]]

    if not best:
        for line in context.split("\n"):
            if len(line) > 80:
                best = [line]
                break

    intro = ""
    if risk_scores:
        high = [d for d, s in risk_scores.items() if s > 60]
        if high:
            intro = f"Based on your elevated risk for {', '.join(high)}: "

    answer = intro + " ".join(best)
    if len(answer) > 650:
        answer = answer[:650] + "…"

    answer += "\n\n⚕️ Always consult a qualified doctor for personalised advice."
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def chat(
    message:     str,
    diseases:    list = None,
    risk_scores: dict = None,
    history:     list = None,
) -> dict:
    """
    Main entry point called by /chat endpoint.
    Returns {"reply": str, "sources": list[str], "mode": str}
    """
    if not message or not message.strip():
        return {"reply": "Please ask a health question.", "sources": [], "mode": "error"}

    # 1. Retrieve relevant context
    context, sources = _retrieve(
        query=message,
        diseases=diseases or [],
        risk_scores=risk_scores or {},
        n=4
    )

    # 2. Try LLM generation
    reply = _generate(message, context, risk_scores or {})
    mode  = "llm"

    # 3. Fallback to retrieval answer
    if not reply:
        reply = _retrieval_answer(message, context, risk_scores or {})
        mode  = "retrieval"

    # 4. Append safety note for medication queries
    med_words = ["medicine","drug","tablet","dose","dosage",
                 "inject","surgery","treatment","cure","prescription"]
    if any(w in message.lower() for w in med_words):
        if "consult" not in reply.lower():
            reply += "\n\n⚠️ Medication and treatment decisions must always involve a registered doctor."

    return {"reply": reply, "sources": sources[:3], "mode": mode}


def init_chatbot():
    """Call once at app startup."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("[Chatbot] Vector store missing — building now...")
        build_vector_store()
    else:
        print("[Chatbot] ✅ Vector store ready")

    # Preload LLM in background thread so first request is fast
    import threading
    threading.Thread(target=_load_generator, daemon=True).start()
