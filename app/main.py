from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag_setup import agent
from app.session_store import load_ctx, save_ctx
from app.memory_setup import load_profile, extract_and_merge_facts, add_turn_to_episodic_memory, current_user_id
from app.auth import verify_google_token, create_session_cookie_value, read_session_cookie_value, COOKIE_NAME, COOKIE_MAX_AGE_SECONDS

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class GoogleLoginRequest(BaseModel):
    credential: str


@app.get("/")
async def index():
    return FileResponse("app/static/index.html")


@app.post("/auth/google")
async def google_login(req: GoogleLoginRequest, response: Response):
    user = verify_google_token(req.credential)
    cookie_value = create_session_cookie_value(user)
    response.set_cookie(
        COOKIE_NAME, cookie_value,
        max_age=COOKIE_MAX_AGE_SECONDS, httponly=True, secure=True, samesite="lax",
    )
    return user


@app.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/me")
async def me(request: Request):
    cookie = request.cookies.get(COOKIE_NAME)
    user = read_session_cookie_value(cookie) if cookie else None
    if user:
        return {"logged_in": True, **user}
    return {"logged_in": False}


def resolve_user_id(request: Request, fallback_session_id: str) -> str:
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        user = read_session_cookie_value(cookie)
        if user:
            return user["sub"]
    return fallback_session_id  # anonymous, browser-local fallback


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    user_id = resolve_user_id(request, req.session_id)
    current_user_id.set(user_id)

    ctx, is_new = load_ctx(agent, user_id)

    message = req.message
    if is_new:
        facts = load_profile(user_id)
        if facts:
            facts_block = "\n".join(f"- {f}" for f in facts)
            message = (
                f"[Known facts about the user from previous sessions:\n{facts_block}\n"
                f"Use this silently as context; don't repeat it back unless relevant.]\n\n"
                f"{req.message}"
            )

    result = await agent.run(message, ctx=ctx)
    save_ctx(user_id, ctx)

    background_tasks.add_task(extract_and_merge_facts, user_id, req.message, str(result))
    background_tasks.add_task(add_turn_to_episodic_memory, user_id, req.message, str(result))

    return ChatResponse(response=str(result))