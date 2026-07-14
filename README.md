# RAG Practice

Experiments building Retrieval-Augmented Generation (RAG) pipelines with [LlamaIndex](https://www.llamaindex.ai/) and Gemini.

## `rag_notebook.ipynb`

Core RAG fundamentals, built around the *"How to Build a Career in AI"* eBook. Indexes the PDF with a local embedding model (`BAAI/bge-small-en-v1.5`) and Gemini as the LLM, then experiments with different retrieval strategies: fixed-size chunking vs. semantic chunking, and `BAAI/bge-small` vs. `all-mpnet-base-v2` embeddings. Ends with a small retrieval evaluation harness (hit rate / MRR) to compare how well each configuration surfaces the correct source chunk for a set of test questions.

## `multimodal_rag.ipynb`

Extends RAG beyond plain text. Loads two research papers (a backpropagation paper and the *ReAct* paper) plus several images (a diagram, a poster, etc.) from `knowledge_base/`. Images are captioned with Gemini's vision model (cached to `knowledge_base/.captions.json` to avoid re-calling the API), and those captions are indexed alongside the paper text in a single combined vector index — so one retrieval step can return a mix of text and image results. When an image is retrieved, the real image (not just its caption) is passed back to Gemini for a grounded final answer.

## `chatbot.ipynb`

A tool-using chatbot agent (`FunctionAgent`) with four tools:
- `query_document` — narrow factual Q&A over the `knowledge_base` papers via vector similarity search
- `summarize_document` — whole-document summaries via `SummaryIndex`/`tree_summarize`, for questions that need broad coverage rather than top-k retrieval
- `get_weather` — live weather lookup (Open-Meteo API)
- `web_search` — real-time web search (Tavily API)

The agent decides which tool (if any) a question needs based on tool descriptions in its system prompt.

## Setup

Requires a `.env` file with `GEMINI_API_KEY` and `TAVILY_API_KEY` (see `utils.py`). Persisted vector indexes (`storage/`, `storage_chatbot/`) are rebuilt automatically on first run if missing.
