"""A chat-style web interface for the PDF study assistant (Phase 11).

Ask a question, pick a model and a search mode, and get a grounded answer whose page citations are links into
the PDF. Every PDF link opens in one shared browser window ("survey_pdf"): the first click
opens it, later clicks move that same window to the new page instead of opening another one.

Run from the project root (needs GROQ_API_KEY, and GOOGLE_API_KEY for Gemini, in .env):

    python app.py            # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import logging
import os
import time

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

import rag_utils as rag

logger = logging.getLogger(__name__)

HOST, PORT = "127.0.0.1", 7860
PDF_WINDOW_NAME = "survey_pdf"

# Display name -> how to build the model. Groq models are checked against the key's model list.
MODELS = {
    "Qwen 3.8 27B (Groq)": ("groq", "qwen/qwen3.8-27b", "none"),
    "GPT OSS 120B (Groq)": ("groq", "openai/gpt-oss-120b", "low"),
    "GPT OSS 20B (Groq)": ("groq", "openai/gpt-oss-20b", "low"),
    "Gemini 3.7 Flash (Google, ~20 requests/day)": ("gemini", rag.GEMINI_MODEL, None),
}
DEFAULT_MODEL = "Qwen 3.8 27B (Groq)"  # most faithful citations in the Phase 7-8 comparisons

# Phase 11: the top 5 chunks by meaning plus the top 5 by keywords (BM25) found 73% of the
# evaluation's multi-page gold pages, against 42% for meaning alone.
SEARCH_MODES = {
    "Meaning + keywords (up to 10 passages)": "hybrid",
    "Meaning only (5 passages)": "dense",
}
DEFAULT_SEARCH = "Meaning + keywords (up to 10 passages)"

# Every PDF link is opened by this script in the one named window, which browsers reuse.
HEAD = f"""
<script>
document.addEventListener("click", (event) => {{
  const link = event.target.closest("a[href*='/pdf?page=']");
  if (!link) return;
  event.preventDefault();
  const viewer = window.open(link.href, "{PDF_WINDOW_NAME}");
  if (viewer) viewer.focus();
}}, true);
</script>
"""

CSS = """
.gradio-container { max-width: 860px !important; margin: auto; }
#title { text-align: center; margin-top: 8px; }
#title p { color: var(--body-text-color-subdued); }
"""


def pdf_link(page: int, label: str) -> str:
    """A Markdown link that opens the PDF at a page (the query makes each page a distinct URL)."""
    return f"[{label}](/pdf?page={page}#page={page})"


def link_citations(answer: str) -> str:
    """Turn every citation in the answer ([p. 55], [Source 1 | pp. 55-56], 【p. 55】) into PDF links."""

    def replace(match):
        references = rag.PAGE_REFERENCE.findall(match.group(1))
        if not references:
            return match.group(0)  # a bracket that isn't a page citation
        links = []
        for first, last in references:
            label = f"p. {first}" if not last else f"pp. {first}-{last}"
            links.append(pdf_link(int(first), label))
        return "(" + "; ".join(links) + ")"

    return rag.BRACKETED.sub(replace, answer)


def format_reply(answer: str, sources, model_label: str, search_label: str, seconds: float) -> str:
    """The answer with linked citations, then the source passages the model was given."""
    lines = [link_citations(answer), ""]
    lines.append(f"<sub>{model_label} · {search_label} · {seconds:.1f} s · pages refer to `survey_chapters.pdf`</sub>")
    lines.append("")
    lines.append("**Passages given to the model**" if answer.strip() == rag.NOT_IN_DOCUMENT else "**Sources**")
    for number, chunk in enumerate(sources, start=1):
        pages = chunk.metadata["page_numbers"]
        preview = " ".join(chunk.page_content.split())[:140]
        lines.append(f"{number}. {pdf_link(pages[0], rag.page_reference(pages))}: {preview}…")
    return "\n".join(lines)


class Assistant:
    """Loads both retrievers once and builds one chain per (model, search mode) on first use."""

    def __init__(self):
        rag.load_api_key("GROQ_API_KEY")
        self.stack = rag.load_retriever()
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

    def answer(self, message: str, history, model_label: str, search_label: str) -> str:
        if not message.strip():
            return "Please type a question about the Economic Survey chapters."
        start = time.perf_counter()
        try:
            result = rag.invoke_with_backoff(self.chain_for(model_label, search_label), message)
        except Exception as error:  # show provider errors in the chat instead of crashing the page
            logger.exception("Request failed")
            return f"⚠️ **{model_label} could not answer:** {str(error)[:300]}\n\nTry another model from the list."
        seconds = time.perf_counter() - start
        return format_reply(result["answer"], result["source_documents"], model_label, search_label, seconds)


def build_interface(assistant: Assistant) -> gr.Blocks:
    with gr.Blocks(title="Economic Survey Study Assistant", fill_height=True) as demo:
        gr.Markdown(
            "## Economic Survey Study Assistant\n"
            "Answers come only from five chapters of India's Economic Survey, with page links.",
            elem_id="title",
        )
        with gr.Row():
            model = gr.Dropdown(choices=list(MODELS), value=DEFAULT_MODEL, label="Model")
            search = gr.Dropdown(choices=list(SEARCH_MODES), value=DEFAULT_SEARCH, label="Search")
        gr.ChatInterface(
            fn=assistant.answer,
            additional_inputs=[model, search],
            chatbot=gr.Chatbot(
                placeholder="**Ask anything about the agriculture, services, industry, infrastructure "
                "or climate chapters.**<br>Each question is answered on its own, from the document only.",
                sanitize_html=True,
                show_label=False,
                height="68vh",
            ),
            textbox=gr.Textbox(placeholder="Ask a question…", submit_btn=True, show_label=False),
            examples=[
                ["How many Global Capability Centres does India host?"],
                ["What was India's share in global services trade in 2024?"],
                ["How many greenfield airports have been approved, and how many are operational?"],
            ],
            fill_height=True,
        )
    return demo


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for name in (__name__, "rag_utils"):
        logging.getLogger(name).setLevel(logging.INFO)
    if not rag.DEFAULT_PDF_PATH.exists():
        raise FileNotFoundError(f"{rag.DEFAULT_PDF_PATH} not found. Run `python download_data.py` first.")

    assistant = Assistant()
    server = FastAPI()

    @server.get("/pdf")
    def pdf(page: int = 1):
        # `page` only makes each page's URL distinct, so a reused window reloads and jumps;
        # the browser's PDF viewer reads the #page=N fragment to open at the page.
        return FileResponse(
            rag.DEFAULT_PDF_PATH,
            media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="survey_chapters.pdf"'},
        )

    app = gr.mount_gradio_app(server, build_interface(assistant), path="/", head=HEAD, css=CSS,
                              theme=gr.themes.Soft())
    logger.info("Open http://%s:%d", HOST, PORT)
    uvicorn.run(app, host=os.environ.get("HOST", HOST), port=int(os.environ.get("PORT", PORT)), log_level="warning")


if __name__ == "__main__":
    main()
