"""A chat-style web interface for the PDF study assistant (Phase 11).

Ask a question, pick a model and a search mode, and get a grounded answer whose page citations
are links into the PDF. Every PDF link opens in one shared browser window ("survey_pdf"): the
first click opens it, later clicks move that same window to the new page. Conversations are
kept in memory while the app runs and are cleared when it stops.

The page itself is web/index.html; this file serves it and the JSON API it calls:

    GET  /                  the page
    GET  /api/config        models, search modes, document name
    GET  /api/chats         past conversations (newest first)
    GET  /api/chats/{id}    one conversation's messages
    POST /api/ask           answer a question, adding it to a conversation
    GET  /pdf?page=N        the PDF (the viewer opens it at #page=N)

Run from the project root (needs GROQ_API_KEY, and GOOGLE_API_KEY for Gemini, in .env):

    python app.py            # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import html
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel

import rag_utils as rag

logger = logging.getLogger(__name__)

HOST, PORT = "127.0.0.1", 7860
PAGE_PATH = Path(__file__).resolve().parent / "web" / "index.html"

# Display name -> how to build the model. Groq models are checked against the key's model list.
MODELS = {
    "Qwen 3.8 27B": ("groq", "qwen/qwen3.8-27b", "none"),
    "GPT OSS 120B": ("groq", "openai/gpt-oss-120b", "low"),
    "GPT OSS 20B": ("groq", "openai/gpt-oss-20b", "low"),
    "Gemini 3.7 Flash": ("gemini", rag.GEMINI_MODEL, None),
}
DEFAULT_MODEL = "Qwen 3.8 27B"  # most faithful citations in the Phase 7-8 comparisons

# Phase 11: the top 5 chunks by meaning plus the top 5 by keywords (BM25) found 73% of the
# evaluation's multi-page gold pages, against 42% for meaning alone.
SEARCH_MODES = {
    "Meaning + keywords": "hybrid",
    "Meaning only": "dense",
}
DEFAULT_SEARCH = "Meaning + keywords"

MARKDOWN = MarkdownIt("commonmark", {"html": False})  # raw HTML in an answer is shown as text, not run


def pdf_href(page: int) -> str:
    """The PDF at a page; the query makes each page a distinct URL, so a reused window reloads."""
    return f"/pdf?page={page}#page={page}"


def link_citations(answer: str) -> str:
    """Turn every citation in the answer ([p. 55], [Source 1 | pp. 55-56], 【p. 55】) into Markdown links."""

    def replace(match):
        references = rag.PAGE_REFERENCE.findall(match.group(1))
        if not references:
            return match.group(0)  # a bracket that isn't a page citation
        links = []
        for first, last in references:
            label = f"p. {first}" if not last else f"pp. {first}-{last}"
            links.append(f"[{label}]({pdf_href(int(first))})")
        return "(" + "; ".join(links) + ")"

    return rag.BRACKETED.sub(replace, answer)


def assistant_message(answer: str, sources, meta: str) -> dict:
    """An answer as the page shows it: HTML with linked citations, plus the passages used."""
    return {
        "role": "assistant",
        "html": MARKDOWN.render(link_citations(answer)),
        "refused": answer.strip() == rag.NOT_IN_DOCUMENT,
        "meta": meta,
        "sources": [
            {
                "label": rag.page_reference(chunk.metadata["page_numbers"]),
                "href": pdf_href(chunk.metadata["page_numbers"][0]),
                "preview": " ".join(chunk.page_content.split())[:160],
            }
            for chunk in sources
        ],
    }


def error_message(text: str) -> dict:
    return {"role": "assistant", "html": f"<p>⚠️ {html.escape(text)}</p>", "refused": False, "meta": "", "sources": []}


class ChatStore:
    """Conversations kept in memory: they last while the app runs and vanish when it stops."""

    def __init__(self):
        self._chats: dict[str, dict] = {}
        self._lock = threading.Lock()  # requests are answered in parallel threads

    def get_or_create(self, chat_id: str | None, first_question: str) -> dict:
        with self._lock:
            if chat_id in self._chats:
                return self._chats[chat_id]
            title = first_question if len(first_question) <= 60 else first_question[:57].rstrip() + "…"
            chat = {"id": uuid.uuid4().hex, "title": title, "messages": [], "updated": time.time()}
            self._chats[chat["id"]] = chat
            return chat

    def add(self, chat: dict, *messages: dict) -> None:
        with self._lock:
            chat["messages"].extend(messages)
            chat["updated"] = time.time()

    def summaries(self) -> list[dict]:
        with self._lock:
            chats = sorted(self._chats.values(), key=lambda chat: chat["updated"], reverse=True)
            return [{"id": chat["id"], "title": chat["title"]} for chat in chats]

    def get(self, chat_id: str) -> dict | None:
        with self._lock:
            return self._chats.get(chat_id)


class Assistant:
    """Loads both retrievers once and builds one chain per (model, search mode) on first use."""

    def __init__(self):
        rag.load_api_key("GROQ_API_KEY")
        self.stack = rag.load_retriever(check_vectors=False)  # quick start: texts and metadata only
        self.retrievers = {
            "dense": self.stack.retriever,
            "hybrid": rag.make_hybrid_retriever(self.stack.retriever, rag.make_bm25_retriever(self.stack.chunks)),
        }
        self.chains = {}

    def chain_for(self, model_label: str, search_label: str):
        key = (model_label, search_label)
        if key not in self.chains:
            provider, model_id, reasoning_effort = MODELS[model_label]
            if provider == "groq":
                llm = rag.make_groq_llm(model_id, reasoning_effort)
            else:
                rag.load_api_key("GOOGLE_API_KEY")
                llm = rag.make_gemini_llm(model_id)
            retriever = self.retrievers[SEARCH_MODES[search_label]]
            self.chains[key] = rag.build_rag_chain_with_sources(retriever, llm)
        return self.chains[key]

    def answer(self, question: str, model_label: str, search_label: str) -> dict:
        start = time.perf_counter()
        try:
            result = rag.invoke_with_backoff(self.chain_for(model_label, search_label), question)
        except Exception as error:  # show provider errors in the chat instead of failing the request
            logger.exception("Request failed")
            return error_message(f"{model_label} could not answer: {str(error)[:300]} Try another model.")
        meta = f"{model_label} · {search_label} · {time.perf_counter() - start:.1f} s"
        return assistant_message(result["answer"], result["source_documents"], meta)


class Question(BaseModel):
    question: str
    model: str = DEFAULT_MODEL
    search: str = DEFAULT_SEARCH
    chat_id: str | None = None


def create_app(assistant: Assistant, store: ChatStore) -> FastAPI:
    app = FastAPI(title="PDF Study Assistant")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE_PATH.read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        return {
            "document": rag.DEFAULT_PDF_PATH.name,
            "models": list(MODELS),
            "default_model": DEFAULT_MODEL,
            "search_modes": list(SEARCH_MODES),
            "default_search": DEFAULT_SEARCH,
        }

    @app.get("/api/chats")
    def chats():
        return store.summaries()

    @app.get("/api/chats/{chat_id}")
    def chat(chat_id: str):
        found = store.get(chat_id)
        if found is None:
            raise HTTPException(404, "No such conversation (they are cleared when the app restarts).")
        return {"id": found["id"], "title": found["title"], "messages": found["messages"]}

    @app.post("/api/ask")
    def ask(request: Question):  # a plain def: FastAPI runs it in a worker thread while the LLM answers
        question = request.question.strip()
        if not question:
            raise HTTPException(400, "Please type a question.")
        if request.model not in MODELS or request.search not in SEARCH_MODES:
            raise HTTPException(400, "Unknown model or search mode.")
        chat = store.get_or_create(request.chat_id, question)
        reply = assistant.answer(question, request.model, request.search)
        store.add(chat, {"role": "user", "text": question}, reply)
        return {"chat_id": chat["id"], "title": chat["title"], "reply": reply}

    @app.get("/pdf")
    def pdf(page: int = 1):
        # `page` only makes each page's URL distinct, so a reused window reloads and jumps;
        # the browser's PDF viewer reads the #page=N fragment to open at the page.
        return FileResponse(
            rag.DEFAULT_PDF_PATH,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{rag.DEFAULT_PDF_PATH.name}"'},
        )

    return app


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in (__name__, "rag_utils"):
        logging.getLogger(name).setLevel(logging.INFO)
    if not rag.DEFAULT_PDF_PATH.exists():
        raise FileNotFoundError(f"{rag.DEFAULT_PDF_PATH} not found. Run `python download_data.py` first.")

    app = create_app(Assistant(), ChatStore())
    host, port = os.environ.get("HOST", HOST), int(os.environ.get("PORT", PORT))
    logger.info("Open http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
