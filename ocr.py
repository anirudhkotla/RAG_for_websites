import requests
from io import BytesIO
from PIL import Image
import pytesseract

def extract_text_from_image(url):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    return text.strip()