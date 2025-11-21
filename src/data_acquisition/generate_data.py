import os
import requests
from config.settings import PERPLEXITY_API_KEY

api_key = PERPLEXITY_API_KEY
url = "https://api.perplexity.ai/chat/completions"
payload = {
    "prompt": "A futuristic cityscape at sunset, ultra-realistic.",
    "model": "image-alpha-001"
}
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Check for API errors before parsing JSON
if response.status_code != 200:
    print("Image generation API failed:", response.status_code, response.text)
    raise RuntimeError("Image generation API call failed")

# Safely get image_url key
json_response = response.json()
if 'image_url' not in json_response:
    print("No image_url in response. Full response:", json_response)
    raise KeyError("image_url missing in API response.")

image_url = json_response['image_url']

folder_name = "data/raw"   # Relative path (recommended)
image_filename = "output_image.png"

os.makedirs(folder_name, exist_ok=True)

# Download the image
img_response = requests.get(image_url)
if img_response.status_code == 200:
    file_path = os.path.join(folder_name, image_filename)
    with open(file_path, 'wb') as f:
        f.write(img_response.content)
    print(f"Image saved to {file_path}")
else:
    print("Failed to download the image. Status:", img_response.status_code, img_response.text)
