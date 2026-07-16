# RAG Practice

**Live app:** [ragagent.fly.dev](https://ragagent.fly.dev/)

Experiments building Retrieval-Augmented Generation (RAG) pipelines with [LlamaIndex](https://www.llamaindex.ai/), plus a deployable chatbot built on top of them. The chatbot can answer questions about two research papers, check the weather, search the web, and it remembers things about you across visits and devices, not just within one chat.

## Layout

```
notebooks/    exploratory RAG notebooks (see below)
app/          the deployable FastAPI chatbot (see below)
knowledge_base/  shared PDFs and images used across notebooks and the app
scripts/      setup scripts (bootstrap.ps1)
```

## `notebooks/rag_notebook.ipynb`

Covers the basics of RAG, built around the *"How to Build a Career in AI"* eBook. It indexes the PDF with a local embedding model (`BAAI/bge-small-en-v1.5`) and uses Gemini as the LLM. Then it tries out different retrieval strategies: fixed-size chunking vs. semantic chunking, and `BAAI/bge-small` vs. `all-mpnet-base-v2` embeddings. It ends with a small retrieval evaluation harness (hit rate / MRR) that compares how well each setup finds the right source chunk for a set of test questions.

## `notebooks/multimodal_rag.ipynb`

Extends RAG beyond plain text. It loads two research papers (a backpropagation paper and the *ReAct* paper) plus several images (a diagram, a poster, etc.) from `knowledge_base/`. Images are captioned with Gemini's vision model, and the captions are cached to `knowledge_base/.captions.json` so the API isn't called again each run. Those captions are indexed alongside the paper text in one combined vector index, so a single retrieval step can return a mix of text and image results. When an image comes back as a result, the real image (not just its caption) is passed to Gemini so the final answer is grounded in what the image actually shows.

## `chatbot.ipynb`

The original prototyping notebook for the tool-using chatbot agent (`FunctionAgent`). It lives at the project root because it shares its persisted index (`storage_chatbot/`) and session files with the deployed app below. New agent behavior gets tried out here first, then ported into `app/`.

## `app/` - the deployable chatbot

The production version of `chatbot.ipynb`. It's a FastAPI app with a chat UI, backed by Groq (`openai/gpt-oss-20b`) instead of Gemini for faster, cheaper inference.

Tools:
- `query_document` - answers narrow factual questions about the `knowledge_base` papers using vector similarity search
- `summarize_backprop_paper` / `summarize_reasoning_paper` - whole document summaries via `SummaryIndex`, one tool per paper so there's no ambiguity about which document is meant
- `get_weather` - live weather lookup (Open-Meteo API)
- `web_search` - real time web search (Tavily API)
- `recall_past_conversation` - searches a memory of every past turn, across all sessions, for the currently logged in user

The agent decides which tool a question needs, if any, based on the tool descriptions in its system prompt.

### Memory that actually persists

This is the part that makes it more than a normal chatbot demo: it remembers you, not just within one conversation, but across visits, days, and devices.

- **Episodic memory** - every message you send and every reply you get is saved into a searchable index (`storage_episodic/`). If you ask something like "what did we talk about before," the agent can search through everything you've said in any past session and pull up the relevant parts.
- **Semantic memory** - after each message, a background task quietly pulls out durable facts about you (your name, what you're working on, your preferences) and saves them to a small profile. The next time you start a new conversation, those facts get fed to the agent automatically, so it already knows you without you repeating yourself.
- **Session memory** - your ongoing conversation itself (the full back and forth) is saved to disk after every message, so closing the tab, restarting your phone, or the server restarting doesn't erase it. You pick up right where you left off.

All three are backed by a persistent disk volume on the server, described more in the deployment section below.

### Google login, so memory follows your account

You can log in with your Google account using the "Sign in" button in the top left of the page. Once logged in, all three kinds of memory above (episodic, semantic, and session) get tied to your actual Google account instead of a random ID stored in your browser. That means your memory follows you between different browsers and different devices, not just the one browser you first used. Your name and profile picture show up next to "Logged in as" once you're signed in, and there's a "Log out" button right next to it.

If you don't log in, the app still works and still remembers you, just tied to that one browser instead of your account, using a random ID saved in local storage.

Note that since this app is a personal project, Google login is currently limited to a manually approved list of test accounts unless the app has been published to production in Google Cloud Console. Anyone not on that list, or logging in after publishing, can still sign in as long as the app requests only basic profile information, which is all it asks for here.

### Running locally

```bash
conda activate rag
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`.

### Deploying (Fly.io)

Ships as a Docker container with a persistent volume mounted at `/data`, so `sessions/`, `storage_episodic/`, `memory/`, and `storage_chatbot/` all survive restarts and redeploys. Most free-tier PaaS options wipe local disk on every sleep/wake cycle, so this matters a lot for an app whose main feature is memory that lasts.

```bash
fly launch --no-deploy
fly volumes create chatbot_data --size 1 --region <region>
fly secrets set GROQ_API_KEY=your_key_here TAVILY_API_KEY=your_key_here GOOGLE_CLIENT_ID=your_client_id_here SESSION_SECRET_KEY=your_generated_secret_here
fly deploy
```
A GitHub Actions workflow (`.github/workflows/fly-deploy.yml`) redeploys automatically on every push to `main`, given a `FLY_API_TOKEN` repo secret.

## Setup

Requires a `.env` file with `GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_CLIENT_ID`, and `SESSION_SECRET_KEY` (see `utils.py`). Gemini is used by the two exploratory notebooks, Groq by the chatbot and app, and the last two are needed for Google login. Persisted vector indexes (`storage/`, `storage_chatbot/`, `storage_episodic/`) are rebuilt automatically on first run if missing.
