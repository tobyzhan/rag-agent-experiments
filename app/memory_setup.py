import json
import os
from datetime import datetime, timezone

import contextvars

current_user_id = contextvars.ContextVar("current_user_id", default="anonymous")


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


from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

def add_turn_to_episodic_memory(session_id: str, user_message: str, response: str) -> None:
    text = f"User: {user_message}\nAssistant: {response}"
    doc = Document(
        text=text,
        metadata={
            "session_id": session_id,
            "user_id": current_user_id.get(),
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
        filters = MetadataFilters(filters=[ExactMatchFilter(key="user_id", value=current_user_id.get())])
        engine = episodic_index.as_query_engine(similarity_top_k=3, response_mode="compact", filters=filters)
        text = str(engine.query(query)).strip()
        return text if text else "No relevant past conversation found."
    except Exception:
        return "No relevant past conversation found."


# --- Semantic memory: distilled, durable facts about the user ---


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

def _profile_path(user_id: str) -> str:
    return os.path.join(DATA_DIR, "memory", f"{user_id}.json")


def load_profile(user_id: str) -> list[str]:
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_profile(user_id: str, facts: list[str]) -> None:
    path = _profile_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def extract_and_merge_facts(user_id: str, user_message: str, response: str) -> None:
    existing = load_profile(user_id)
    prompt = EXTRACTION_PROMPT.format(
        existing_facts=json.dumps(existing),
        user_message=user_message,
        response=response,
    )
    try:
        result = Settings.llm.complete(prompt)
        facts = json.loads(str(result).strip())
        if isinstance(facts, list):
            save_profile(user_id, facts)
    except Exception:
        pass