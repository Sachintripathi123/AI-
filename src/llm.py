import os
from dotenv import load_dotenv
from google import genai
#from langchain_groq import ChatGroq

load_dotenv()


def create_llm():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set in .env"
        )

    client = genai.Client( 
       

        api_key=api_key
    )

    return client