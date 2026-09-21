# Personal PDF Study Assistant (RAG)

A retrieval-augmented generation pipeline over prose chapters of the India Economic
Survey, built from scratch with LangChain (LCEL), local `all-MiniLM-L6-v2` embeddings,
FAISS, and Gemini Flash.

> **Status: Phase 0 (environment).** This README is a placeholder. The full write-up —
> architecture diagram, retrieval evaluation table, documented failure modes, and the
> "Why RAG instead of long context?" discussion — is written in Phase 10.

## Setup so far

```bash
conda create -n rag python=3.12 -y
conda activate rag
pip install -r requirements.txt

cp .env.example .env      # then paste your key from https://aistudio.google.com/apikey
python check_setup.py     # verifies GPU, lists live Gemini models, makes one test call
```

## License

MIT (added in Phase 10).
