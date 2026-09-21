"""Phase 0 acceptance check.

Run this once after installing dependencies and creating your `.env`:

    conda activate rag
    python check_setup.py

It answers three questions, in order:

  1. Did PyTorch find the GPU?  (embeddings run locally in Phase 5)
  2. Which Gemini models does *this key* have access to right now?
  3. Does one minimal end-to-end call through LangChain actually return text?

A note on why step 3 is not redundant with step 2: Google's ListModels endpoint
happily advertises models that have since been retired. On this key it listed
`gemini-2.5-flash`, which then returned 404 "no longer available" on first use.
So we do not merely read the catalogue -- we call each candidate until one
actually answers, and report the survivor.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

# The Gemini REST endpoint that lists every model visible to a given key.
MODELS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Model IDs containing any of these are snapshots or specialities we do not want:
# preview builds get retired without notice, and the rest are not chat models.
EXCLUDED_MARKERS = (
    "preview", "exp", "thinking", "image", "tts", "audio",
    "robotics", "computer-use", "deep-research", "banana", "lyria",
)


def section(title: str) -> None:
    """Print a visually obvious separator so the checks are easy to tell apart."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_torch() -> None:
    """Report whether PyTorch can see the NVIDIA GPU, and on which architecture."""
    section("1. PyTorch / CUDA")

    import torch

    print(f"torch version      : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n  -> No GPU visible. This is NOT fatal. The embedding model is small")
        print("     (22M parameters); it will just run slower on CPU.")
        return

    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"compute capability : sm_{major}{minor}")

    # Availability is not the same as usability. A wheel built without kernels for
    # this GPU's architecture reports `is_available() == True` and then fails on
    # the first real operation with "no kernel image is available for execution".
    # So we force an actual computation and synchronise to surface any such error.
    try:
        left = torch.randn(512, 512, device="cuda")
        right = torch.randn(512, 512, device="cuda")
        product = left @ right
        torch.cuda.synchronize()
        print(f"real matmul on GPU : OK (result sum = {product.sum().item():.2f})")
    except Exception as error:  # noqa: BLE001 - report any failure verbatim
        print(f"real matmul on GPU : FAILED -> {type(error).__name__}: {error}")
        print("\n  -> The wheel lacks kernels for this GPU. Embeddings will need CPU.")


def list_chat_models(api_key: str) -> list[str]:
    """Ask Google which models this key can use; return the chat-capable ones."""
    section("2. Gemini models advertised to this key")

    request = urllib.request.Request(
        f"{MODELS_ENDPOINT}?key={api_key}&pageSize=200",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code} from Google. Response body:\n{body}")
        print("\n  -> 400 usually means a malformed key string;")
        print("     403 means the key is valid but the API is not enabled for it.")
        sys.exit(1)

    chat_models: list[str] = []
    for model in payload.get("models", []):
        # The API returns qualified names like "models/gemini-3.5-flash".
        model_id = model["name"].removeprefix("models/")
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue  # embedding / TTS / image endpoints -- not usable as a chat LLM
        chat_models.append(model_id)

    print(f"{len(chat_models)} chat-capable models advertised.")
    return chat_models


def version_of(model_id: str) -> float:
    """Extract the numeric version from a model ID, for ranking newest-first.

    'gemini-3.5-flash' -> 3.5 ; 'gemini-flash-latest' -> -1.0

    The moving aliases ('-latest') sort last on purpose. They are convenient but
    silently change what they point at, which is the opposite of what a repo
    meant to be reproducible six months from now needs.
    """
    match = re.search(r"gemini-(\d+(?:\.\d+)?)-", model_id)
    return float(match.group(1)) if match else -1.0


def rank_flash_candidates(chat_models: list[str]) -> list[str]:
    """Return Flash-tier candidates, best first.

    The brief requires a Flash model: it carries the free tier's highest request
    quota and is fast enough to keep Phase 8's spot checks cheap. We rank rather
    than pick one, because the top choice may turn out to be a retired listing.
    """
    candidates = [
        name for name in chat_models
        if "flash" in name and not any(tag in name for tag in EXCLUDED_MARKERS)
    ]
    # Prefer full Flash over Flash-Lite: lite trades answer quality for
    # throughput, and our bottleneck is quality, not requests per second.
    full = sorted((n for n in candidates if "lite" not in n), key=version_of, reverse=True)
    lite = sorted((n for n in candidates if "lite" in n), key=version_of, reverse=True)
    ranked = full + lite

    print("\nFlash-tier candidates, best first:")
    for name in ranked:
        print(f"  {name}")
    return ranked


def try_model(model_id: str) -> tuple[bool, str]:
    """Attempt one minimal call. Returns (succeeded, message)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    # temperature=0 makes the model pick the most likely token every time, so
    # repeated runs are reproducible -- we want determinism, not variety.
    #
    # thinking_budget=0 switches off the model's internal reasoning pass. On
    # Gemini 3.5 Flash this cut a one-line answer from 289 output tokens to 17.
    # For extractive RAG the model is copying facts out of retrieved text, not
    # solving anything, so the reasoning pass buys nothing and spends quota.
    # Models that ignore a zero budget simply keep thinking; that is not an error.
    llm = ChatGoogleGenerativeAI(model=model_id, temperature=0, thinking_budget=0)

    try:
        response = llm.invoke("Reply with exactly: RAG setup OK")
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {str(error)[:120]}"

    usage = response.usage_metadata or {}
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)

    # response.text is the flattened string. In LangChain 1.x, `.content` is a
    # list of typed content blocks, so reach for `.text` when you want a string.
    return True, (
        f"answer={response.text!r} "
        f"output_tokens={usage.get('output_tokens')} reasoning_tokens={reasoning}"
    )


def main() -> None:
    # load_dotenv reads the `.env` file next to this script and copies each
    # KEY=value line into the process environment, so os.environ can see it.
    # The key therefore never appears in source code or in shell history.
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    if not api_key or api_key == "your-key-here":
        print(f"GOOGLE_API_KEY is missing from {env_path}")
        print("Copy .env.example to .env and paste your key from")
        print("https://aistudio.google.com/apikey")
        sys.exit(1)

    # Print only the shape of the key, never the key itself -- this output tends
    # to get pasted into chats, issues and screenshots.
    print(f".env loaded. GOOGLE_API_KEY present ({len(api_key)} chars, "
          f"starts with {api_key[:4]}...).")

    check_torch()
    ranked = rank_flash_candidates(list_chat_models(api_key))

    section("3. Calling each candidate until one actually answers")
    chosen = None
    for model_id in ranked:
        succeeded, message = try_model(model_id)
        print(f"  {model_id:<32} {'OK  ' if succeeded else 'DEAD'} {message}")
        if succeeded and chosen is None:
            chosen = model_id
            break

    if chosen is None:
        print("\nNo Flash model responded. Check quota at aistudio.google.com.")
        sys.exit(1)

    section("Phase 0 complete")
    print(f"Record this model ID for Phase 7:  {chosen}")


if __name__ == "__main__":
    main()
