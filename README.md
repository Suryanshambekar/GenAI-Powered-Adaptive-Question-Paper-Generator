## GenAI-Powered Adaptive Question Paper Generator

### Overview
This project turns course PDFs into curated exam-style questions by combining retrieval-augmented generation (RAG) with a fine-tuned T5 question-generation model. A Streamlit interface orchestrates PDF ingestion, semantic chunking, FAISS-based retrieval, and guided question synthesis to keep the prompts aligned with the uploaded material.

### The project is deployed on:

Streamlit Cloud: [https://genai-powered-question-paper-generator.streamlit.app](https://genai-powered-question-paper-generator.streamlit.app)

### Features
- Adaptive question generation from any lecture PDF without manual curation.
- Lightweight SentenceTransformer embeddings (`all-MiniLM-L6-v2`) for fast semantic similarity search.
- FAISS index construction per upload to ensure questions stay context-aware.
- Hugging Face T5 model fine-tuned for question generation and packaged locally for offline use.
- Streamlit UI with PDF upload, progress reporting, and incremental question display.

### Repository Structure
- `streamlit_app.py` – End-to-end Streamlit workflow covering PDF parsing, chunking, retrieval, and question generation.
- `GenAI_Project_Draft_2(RAG+HF).ipynb` – Notebook draft capturing experimentation with the RAG pipeline and Hugging Face model.
- `qg_t5_model/` – Local T5 question-generation checkpoint plus tokenizer artifacts (tracked via Git LFS).
- `requirements.txt` – Python dependencies used across the notebook and Streamlit app.

### Methodology
1. **Document Ingestion** – PDFs are parsed with PyPDF2 and concatenated into raw text.
2. **Chunking Strategy** – Text is split into ~300-word windows to balance semantic coherence and retrieval granularity.
3. **Embedding & Indexing** – Each chunk is embedded with `all-MiniLM-L6-v2` and inserted into an in-memory FAISS `IndexFlatL2`.
4. **Context Retrieval** – For every chunk under consideration, the two most similar chunks are retrieved and merged to enrich context.
5. **Question Generation** – A T5 question-generation model (`qg_t5_model`) consumes the merged context using a controlled decoding configuration (beam search, temperature, top-p) to yield focused questions.
6. **Adaptive Sampling** – Generation iterates over chunks until the user-requested number of questions is satisfied, preventing redundant prompts.

### Technology Stack
- Streamlit for interactive UX.
- PyPDF2 for document parsing.
- SentenceTransformers + FAISS for dense retrieval.
- Hugging Face Transformers (T5) for sequence-to-sequence generation.
- PyTorch as the underlying deep learning runtime.

### Future Enhancements
- Add answer generation to pair each question with concise reference answers.
- Introduce difficulty tagging by leveraging Bloom-level classifiers on generated questions.
- Persist FAISS indices for large corpora to support multi-document assessments.

