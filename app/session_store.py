import json
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context, JsonSerializer

DATA_DIR = os.environ.get("DATA_DIR", ".")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_ctx(agent: FunctionAgent, session_id: str) -> tuple[Context, bool]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return Context(agent), True

    with open(path, encoding="utf-8") as f:
        ctx_dict = json.load(f)
    return Context.from_dict(agent, ctx_dict, serializer=JsonSerializer()), False


def save_ctx(session_id: str, ctx: Context) -> None:
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(ctx_dict, f)