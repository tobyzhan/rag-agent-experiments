# RAG Practice

Experiments building Retrieval-Augmented Generation (RAG) pipelines with [LlamaIndex](https://www.llamaindex.ai/), plus a deployable tool-using chatbot built on top of them.

## Layout

```
notebooks/    exploratory RAG notebooks (see below)
app/          deployable FastAPI chatbot (see below)
knowledge_base/  shared PDFs/images used across notebooks and the app
scripts/      setup scripts (bootstrap.ps1)
```

## `notebooks/rag_notebook.ipynb`

Core RAG fundamentals, built around the *"How to Build a Career in AI"* eBook. Indexes the PDF with a local embedding model (`BAAI/bge-small-en-v1.5`) and Gemini as the LLM, then experiments with different retrieval strategies: fixed-size chunking vs. semantic chunking, and `BAAI/bge-small` vs. `all-mpnet-base-v2` embeddings. Ends with a small retrieval evaluation harness (hit rate / MRR) to compare how well each configuration surfaces the correct source chunk for a set of test questions.

## `notebooks/multimodal_rag.ipynb`

Extends RAG beyond plain text. Loads two research papers (a backpropagation paper and the *ReAct* paper) plus several images (a diagram, a poster, etc.) from `knowledge_base/`. Images are captioned with Gemini's vision model (cached to `knowledge_base/.captions.json` to avoid re-calling the API), and those captions are indexed alongside the paper text in a single combined vector index — so one retrieval step can return a mix of text and image results. When an image is retrieved, the real image (not just its caption) is passed back to Gemini for a grounded final answer.

## `chatbot.ipynb`

The original prototyping notebook for the tool-using chatbot agent (`FunctionAgent`), kept at the project root since it shares its persisted index (`storage_chatbot/`) and session files with the deployed app below. This is where new agent behavior gets tried out before it's ported into `app/`.

## `app/` — deployable chatbot

The productionized version of `chatbot.ipynb`: a FastAPI app with a chat UI, backed by Groq (`openai/gpt-oss-20b`) instead of Gemini for faster, cheaper inference. Tools:
- `query_document` — narrow factual Q&A over the `knowledge_base` papers via vector similarity search
- `summarize_backprop_paper` / `summarize_reasoning_paper` — whole-document summaries via `SummaryIndex`, one tool per paper to avoid ambiguity about which document is meant
- `get_weather` — live weather lookup (Open-Meteo API)
- `web_search` — real-time web search (Tavily API)
- `recall_past_conversation` — searches an episodic memory index built from every past turn, across all sessions

The agent decides which tool (if any) a question needs based on tool descriptions in its system prompt.

**Memory** is two-layered:
- *Episodic* — every turn is embedded and stored in a searchable vector index (`storage_episodic/`), so the agent can look up specific past exchanges from any session.
- *Semantic* — after each turn, a background task extracts durable facts about the user (name, ongoing projects, preferences) into `memory/user_profile.json`. These are silently injected into the system prompt at the start of every new session, so the agent already "knows" you without re-stating anything.

Per-session conversation state (`sessions/`) persists across server restarts via LlamaIndex's `Context` serialization, so a session survives a page refresh or a server redeploy.

### Running locally

```bash
conda activate rag
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`.

### Deploying (Fly.io)

Ships as a Docker container with a persistent volume mounted at `/data`, so `sessions/`, `storage_episodic/`, `memory/`, and `storage_chatbot/` all survive restarts and redeploys — unlike most free-tier PaaS options, which wipe local disk on every sleep/wake cycle.

```bash
fly launch --no-deploy
fly volumes create chatbot_data --size 1 --region <region>
fly secrets set GROQ_API_KEY=your_key_here TAVILY_API_KEY=your_key_here
fly deploy
```
A GitHub Actions workflow (`.github/workflows/fly-deploy.yml`) redeploys automatically on every push to `main`, given a `FLY_API_TOKEN` repo secret.

## Setup

Requires a `.env` file with `GEMINI_API_KEY`, `GROQ_API_KEY`, and `TAVILY_API_KEY` (see `utils.py`) — Gemini is used by the two exploratory notebooks, Groq by the chatbot/app. Persisted vector indexes (`storage/`, `storage_chatbot/`, `storage_episodic/`) are rebuilt automatically on first run if missing.
