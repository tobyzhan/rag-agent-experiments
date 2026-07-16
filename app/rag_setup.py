import os

import fitz
import requests
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from tavily import TavilyClient

import utils

DATA_DIR = os.environ.get("DATA_DIR", ".")
STORAGE_DIR = os.path.join(DATA_DIR, "storage_chatbot")
DOCS_DIR = "./knowledge_base"   # source PDFs ship with the image, not on the volume

Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=utils.get_groq_api_key())
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from app.memory_setup import recall_past_conversation  # after Settings are configured


def load_pdf_as_document(path: str) -> Document:
    pdf = fitz.open(path)
    text = "\n\n".join(page.get_text() for page in pdf)
    return Document(text=text, metadata={"file_name": os.path.basename(path)})


def build_index_if_missing() -> VectorStoreIndex:
    if os.path.exists(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)

    documents = [
        load_pdf_as_document(os.path.join(DOCS_DIR, "learning_representations_by_backprop.pdf")),
        load_pdf_as_document(os.path.join(DOCS_DIR, "synergizing_reasonsing_llm.pdf")),
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index


index = build_index_if_missing()
query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

summary_indices = {
    "backprop": SummaryIndex.from_documents(
        [load_pdf_as_document(os.path.join(DOCS_DIR, "learning_representations_by_backprop.pdf"))]
    ),
    "reasoning": SummaryIndex.from_documents(
        [load_pdf_as_document(os.path.join(DOCS_DIR, "synergizing_reasonsing_llm.pdf"))]
    ),
}
summary_engines = {k: v.as_query_engine(response_mode="compact") for k, v in summary_indices.items()}


def query_document(question: str) -> str:
    """Answer a question using the loaded documents: a paper on backpropagation
    (learning_representations_by_backprop.pdf) and a paper on synergizing reasoning and acting in LLMs
    (synergizing_reasonsing_llm.pdf). Use this for any question about the
    content of either paper, not for general knowledge or real-time info."""
    return str(query_engine.query(question))


def summarize_backprop_paper(question: str) -> str:
    """Summarize or answer overview questions about the backpropagation paper
    (learning_representations_by_backprop.pdf)."""
    return str(summary_engines["backprop"].query(question))


def summarize_reasoning_paper(question: str) -> str:
    """Summarize or answer overview questions about the ReAct/reasoning-acting paper
    (synergizing_reasonsing_llm.pdf)."""
    return str(summary_engines["reasoning"].query(question))


def get_weather(location: str) -> str:
    """Get the current weather for a given location (city name).
    Use this for any question about current weather conditions or forecast."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()

    if not geo.get("results"):
        return f"Could not find location: {location}"

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    resolved_name = geo["results"][0]["name"]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
        },
        timeout=10,
    ).json()

    current = weather.get("current", {})
    return (
        f"Current weather in {resolved_name}: "
        f"{current.get('temperature_2m')}F, "
        f"wind {current.get('wind_speed_10m')} mph."
    )


tavily_client = TavilyClient(api_key=utils.get_tavily_api_key())


def web_search(query: str) -> str:
    """Search the web for current, real-time, or recent information
    (news, scores, prices, events, anything time-sensitive). Use this
    for anything that requires up-to-date info you wouldn't know from
    training data or from the loaded document."""
    results = tavily_client.search(query=query, max_results=3)
    snippets = [r["content"] for r in results.get("results", [])]
    return "\n\n".join(snippets) if snippets else "No results found."


tools = [
    FunctionTool.from_defaults(fn=query_document),
    FunctionTool.from_defaults(fn=summarize_backprop_paper),
    FunctionTool.from_defaults(fn=summarize_reasoning_paper),
    FunctionTool.from_defaults(fn=get_weather),
    FunctionTool.from_defaults(fn=web_search),
    FunctionTool.from_defaults(fn=recall_past_conversation),
]

agent = FunctionAgent(
    tools=tools,
    llm=Settings.llm,
    streaming=False,
    allow_parallel_tool_calls=False,
    system_prompt=(
    "You are a helpful assistant with these capabilities: "
    "answering general questions directly, answering narrow factual questions about "
    "loaded documents via query_document, answering whole-document summary/overview "
    "requests via summarize_backprop_paper or summarize_reasoning_paper, and answering "
    "real-time questions via get_weather or web_search. "
    "Only call a tool when the question genuinely requires it. "
    "If the user asks about something discussed earlier in THIS conversation, answer "
    "directly from the chat history you already have - do not claim you have no record "
    "of it. Only use recall_past_conversation when the user references a DIFFERENT, "
    "earlier session whose content is not already visible in this conversation. "
    "When a tool returns an answer, present that answer directly to the user as "
    "your response. Do not comment on, evaluate, or praise the tool's output as if "
    "reviewing someone else's work - the tool's result IS your answer, just relay it "
    "clearly (lightly rephrased for readability if needed). "
    "For plain general-knowledge or conversational questions, just answer directly."
),
)