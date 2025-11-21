import os
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if PERPLEXITY_API_KEY is None:
    raise ValueError("API key is missing. Make sure it's in your .env file and named PERPLEXITY_API_KEY")
