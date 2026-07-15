import json
import os
from datetime import datetime, timezone

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# --- Episodic memory: every past turn, searchable by content ---

DATA_DIR = os.environ.get("DATA_DIR", ".")
EPISODIC_STORAGE_DIR = os.path.join(DATA_DIR, "storage_episodic")

def load_or_create_episodic_index() -> VectorStoreIndex:
    if os.path.exists(EPISODIC_STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=EPISODIC_STORAGE_DIR)
        return load_index_from_storage(storage_context)
    index = VectorStoreIndex.from_documents([])
    index.storage_context.persist(persist_dir=EPISODIC_STORAGE_DIR)
    return index


episodic_index = load_or_create_episodic_index()
episodic_query_engine = episodic_index.as_query_engine(similarity_top_k=3, response_mode="compact")


def add_turn_to_episodic_memory(session_id: str, user_message: str, response: str) -> None:
    text = f"User: {user_message}\nAssistant: {response}"
    doc = Document(
        text=text,
        metadata={
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    episodic_index.insert(doc)
    episodic_index.storage_context.persist(persist_dir=EPISODIC_STORAGE_DIR)


def recall_past_conversation(query: str) -> str:
    """Search across all previous conversations, including from other sessions,
    for relevant past context. Use this when the user references something
    discussed earlier, asks you to recall something from a prior session, or
    when prior context would help answer accurately."""
    try:
        result = episodic_query_engine.query(query)
        text = str(result).strip()
        return text if text else "No relevant past conversation found."
    except Exception:
        return "No relevant past conversation found."


# --- Semantic memory: distilled, durable facts about the user ---

PROFILE_PATH = os.path.join(DATA_DIR, "memory", "user_profile.json")

EXTRACTION_PROMPT = """You maintain a running profile of durable facts about a user, based on their conversation with an assistant.

Existing known facts:
{existing_facts}

New exchange:
User: {user_message}
Assistant: {response}

Return an updated JSON array of facts (strings only). Rules:
- Only include durable facts about the user: their name, ongoing projects, preferences, role, or similar identity/context info.
- Do NOT include one-off trivia, weather lookups, web search results, or anything not about the user themselves.
- Merge/dedupe with existing facts; update a fact if the new exchange changes it (e.g. a new project replacing an old one).
- If nothing new or durable was learned, return the existing facts unchanged.
- Return ONLY a JSON array of strings, nothing else - no markdown, no commentary.
"""


def load_profile() -> list[str]:
    if not os.path.exists(PROFILE_PATH):
        return []
    with open(PROFILE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_profile(facts: list[str]) -> None:
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def extract_and_merge_facts(user_message: str, response: str) -> None:
    existing = load_profile()
    prompt = EXTRACTION_PROMPT.format(
        existing_facts=json.dumps(existing),
        user_message=user_message,
        response=response,
    )
    try:
        result = Settings.llm.complete(prompt)
        facts = json.loads(str(result).strip())
        if isinstance(facts, list):
            save_profile(facts)
    except Exception:
        pass  # best-effort bookkeeping; never break the actual chat turn over this