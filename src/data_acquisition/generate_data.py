import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import base64

# 1. Load Environment Variables
# This forces Python to look for .env in the parent directories if not found in current
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GEMINI_API_KEY")

# DEBUGGING: Print result to see if it worked
if not api_key:
    print("ERROR: API Key is missing! Check your .env file location.")
    # TEMPORARY FIX: If .env still fails, uncomment the line below and paste your key directly to test
    # api_key = "AIzaSy..." 
else:
    print("API Key found successfully.")

client = genai.Client(api_key=api_key)

# 2. Configuration
prompt = "A cute robot gardener watering plants in a greenhouse, isometric view, 3d render style"
folder_name = "data/raw"
count = 4

# IMPORTANT: Create the folder if it doesn't exist yet
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Created missing folder: {folder_name}")

print(f"Generating {count} images...")

# 4. Call the API
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"] # Request an image explicitly
        )
    )

    found_image = False
    if response.candidates:
        for part in response.candidates[0].content.parts:
            # Check if the part contains executable code (sometimes it writes code to draw) 
            # or raw image bytes (inline_data)
            if part.inline_data:
                img_data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                img = Image.open(BytesIO(img_data))
                img.show()
                img.save("gemini_defect_test.png")
                print("✅ Saved gemini_defect_test.png")
                found_image = True
    
    if not found_image:
        print("The model returned text instead of an image. It might have refused the request.")
        print("Response text:", response.text)

except Exception as e:
    print(f"Error: {e}")
    print("\nIf this failed, you MUST enable billing to use the 'imagen' models.")