import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key():
    return os.environ["OPENAI_API_KEY"]

def get_gemini_api_key():
    return os.environ["GEMINI_API_KEY"]

def get_tavily_api_key():
    return os.environ["TAVILY_API_KEY"]