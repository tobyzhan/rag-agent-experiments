import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key():
    return os.environ["OPENAI_API_KEY"]

def get_gemini_api_key():
    return os.environ["GEMINI_API_KEY"]

def get_tavily_api_key():
    return os.environ["TAVILY_API_KEY"]

def get_groq_api_key():
    return os.environ["GROQ_API_KEY"]


def get_google_client_id():
    return os.environ["GOOGLE_CLIENT_ID"]

def get_session_secret_key():
    return os.environ["SESSION_SECRET_KEY"]