# RAG Practice

**Live app: [ragagent.fly.dev](https://ragagent.fly.dev/)**

Experiments building Retrieval-Augmented Generation (RAG) pipelines with [LlamaIndex](https://www.llamaindex.ai/), plus a deployable tool-using chatbot built on top of them.

## Layout

```
notebooks/    exploratory RAG notebooks (see below)
app/          deployable FastAPI chatbot (see below)
knowledge_base/  shared PDFs/images used across notebooks and the app
scripts/      setup scripts (bootstrap.ps1)
```

## `notebooks/rag_notebook.ipynb`

Covers the basics of RAG, built around the *"How to Build a Career in AI"* eBook. It indexes the PDF with a local embedding model (`BAAI/bge-small-en-v1.5`) and uses Gemini as the LLM. Then it tries out different retrieval strategies: fixed-size chunking vs. semantic chunking, and `BAAI/bge-small` vs. `all-mpnet-base-v2` embeddings. It ends with a small retrieval evaluation harness (hit rate / MRR) that compares how well each setup finds the right source chunk for a set of test questions.

## `notebooks/multimodal_rag.ipynb`

Extends RAG beyond plain text. It loads two research papers (a backpropagation paper and the *ReAct* paper) plus several images (a diagram, a poster, etc.) from `knowledge_base/`. Images are captioned with Gemini's vision model, and the captions are cached to `knowledge_base/.captions.json` so the API isn't called again each run. Those captions are indexed alongside the paper text in one combined vector index, so a single retrieval step can return a mix of text and image results. When an image comes back as a result, the real image (not just its caption) is passed to Gemini so the final answer is grounded in what the image actually shows.

## `chatbot.ipynb`

The original prototyping notebook for the tool-using chatbot agent (`FunctionAgent`). It lives at the project root because it shares its persisted index (`storage_chatbot/`) and session files with the deployed app below. New agent behavior gets tried out here first, then ported into `app/`.

## `app/`: deployable chatbot

The production version of `chatbot.ipynb`. It's a FastAPI app with a chat UI, backed by Groq (`openai/gpt-oss-20b`) instead of Gemini for faster, cheaper inference. Try it live at [ragagent.fly.dev](https://ragagent.fly.dev/). Tools:
- `query_document` — narrow factual Q&A over the `knowledge_base` papers via vector similarity search
- `summarize_backprop_paper` / `summarize_reasoning_paper` — whole-document summaries via `SummaryIndex`, one tool per paper so it's clear which document is meant
- `get_weather` — live weather lookup (Open-Meteo API)
- `web_search` — real-time web search (Tavily API)
- `recall_past_conversation` — searches an episodic memory index built from every past turn, across all sessions

The agent picks which tool (if any) a question needs based on the tool descriptions in its system prompt.

### Session-to-session memory

This is the core feature of the app: the agent remembers you across sessions, not just within one conversation. Memory works in two layers:

- **Episodic memory** — every turn you have with the agent is embedded and stored in a searchable vector index (`storage_episodic/`). This means the agent can look back and find specific past exchanges from any session, not just the current one.
- **Semantic memory** — after each turn, a background task pulls out durable facts about you (your name, ongoing projects, preferences) and saves them to `memory/user_profile.json`. These facts are quietly added to the system prompt at the start of every new session, so the agent already knows you when you show up. You never have to repeat yourself.

On top of that, per-session conversation state (`sessions/`) is saved across server restarts using LlamaIndex's `Context` serialization. So a session survives a page refresh or even a server redeploy. Between the episodic index, the semantic profile, and persisted session state, the agent keeps its memory of you intact no matter how much time passes between conversations.

### Running locally

```bash
conda activate rag
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`.

### Deploying (Fly.io)

Ships as a Docker container with a persistent volume mounted at `/data`, so `sessions/`, `storage_episodic/`, `memory/`, and `storage_chatbot/` all survive restarts and redeploys. Most free-tier PaaS options wipe local disk on every sleep/wake cycle, so this matters.

```bash
fly launch --no-deploy
fly volumes create chatbot_data --size 1 --region <region>
fly secrets set GROQ_API_KEY=your_key_here TAVILY_API_KEY=your_key_here
fly deploy
```
A GitHub Actions workflow (`.github/workflows/fly-deploy.yml`) redeploys automatically on every push to `main`, given a `FLY_API_TOKEN` repo secret.

## Setup

Requires a `.env` file with `GEMINI_API_KEY`, `GROQ_API_KEY`, and `TAVILY_API_KEY` (see `utils.py`). Gemini is used by the two exploratory notebooks, Groq by the chatbot/app. Persisted vector indexes (`storage/`, `storage_chatbot/`, `storage_episodic/`) are rebuilt automatically on first run if missing.
