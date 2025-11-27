import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np


# ============= 1. LOAD MODEL + TOKENIZER =================

@st.cache_resource
def load_qg_model():
    model = T5ForConditionalGeneration.from_pretrained(
    "qg_t5_model"
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = T5Tokenizer.from_pretrained("qg_t5_model")
    return model, tokenizer


model, tokenizer = load_qg_model()


# ================ 2. EMBEDDING MODEL ======================

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# ================ 3. PDF → TEXT ===========================

def extract_pdf_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ============= 4. CHUNK TEXT ==============================

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


# ============= 5. BUILD FAISS INDEX =======================

def build_pdf_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings


# =========== 6. RETRIEVAL =============================

def retrieve_similar_chunks(chunk, index, chunks, top_k=2):
    q_emb = embedder.encode([chunk], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]


# ============= 7. QUESTION GENERATION =====================

def generate_question(context):
    prompt = "generate question: " + context
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        tokens,
        max_length=80,
        num_beams=4,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# =========================================================
# 🎯 STREAMLIT UI
# =========================================================

st.title("PDF → Question Generation with RAG + T5")
st.write("Upload a lecture PDF and generate exam-style questions.")


uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

num_q = st.number_input("How many questions to generate?", min_value=1, max_value=50, value=10)

if st.button("Generate Questions"):
    if not uploaded_pdf:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Extracting text..."):
            text = extract_pdf_text(uploaded_pdf)

        chunks = chunk_text(text, chunk_size=300)
        st.write(f"PDF split into {len(chunks)} chunks.")

        # Build FAISS index
        with st.spinner("Building embeddings..."):
            index, emb = build_pdf_index(chunks)

        generated = []
        st.write("Generating questions...")

        for ch in chunks:
            similar = retrieve_similar_chunks(ch, index, chunks, top_k=2)
            merged = ch + " " + " ".join(similar)

            q = generate_question(merged)
            generated.append(q)

            if len(generated) >= num_q:
                break

        st.success("Done!")

        st.write("---")
        st.subheader("Generated Questions")

        for i, q in enumerate(generated, start=1):
            st.write(f"{i}. {q}")
