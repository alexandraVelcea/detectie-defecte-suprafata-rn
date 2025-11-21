import sys
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
module_path = os.getcwd()

sys.path.insert(0, module_path)

print("WORKING DIRECTORY: ", os.getcwd())
print("PTYHONPATH: ", os.environ.get('PYTHONPATH'))
print("SYSPATH: ", sys.path)


from google import genai
from config.settings import GEMINI_API_KEY


def ask_gemini(prompt: str) -> str:
    """
    Send a prompt to Gemini AI and return the model's text response.
    """

    # Create the client
    client = genai.Client()

    models = client.models.list_models()
    for model in models:
        print(model.name, model.supported_generation_methods)

    # Interrogate the model
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    question = "Explain the difference between supervised and unsupervised learning."
    answer = ask_gemini(question)
    print("Gemini response:\n", answer)
