import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("Fetching model list...")
print("-" * 50)

try:
    # Get the list of models
    for model in client.models.list():
        # We print the name attribute which is standard
        # Depending on the version, the ID might be inside 'name' or 'display_name'
        print(f"ID: {model.name}")
        
except Exception as e:
    print(f"Error: {e}")
    # Fallback: inspect the object to help debug
    print("\nCould not list models normally. Here are the attributes of the last model object:")
    if 'model' in locals():
        print(dir(model))