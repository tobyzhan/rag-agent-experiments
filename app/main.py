from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag_setup import agent
from app.session_store import load_ctx, save_ctx
from app.memory_setup import load_profile, extract_and_merge_facts, add_turn_to_episodic_memory

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    ctx, is_new = load_ctx(agent, req.session_id)

    message = req.message
    if is_new:
        facts = load_profile()
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
            message = (
                f"[Known facts about the user from previous sessions:\n{facts_block}\n"
                f"Use this silently as context; don't repeat it back unless relevant.]\n\n"
                f"{req.message}"
            )

    result = await agent.run(message, ctx=ctx)
    save_ctx(req.session_id, ctx)

    background_tasks.add_task(extract_and_merge_facts, req.message, str(result))
    background_tasks.add_task(add_turn_to_episodic_memory, req.session_id, req.message, str(result))

    return ChatResponse(response=str(result))