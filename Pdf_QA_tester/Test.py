from dotenv import load_dotenv
import os

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Accéder aux variables d'environnement
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
google_api_key = os.getenv('GEMINI_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Utiliser les clés API dans votre code
print("Hugging Face API Key:", huggingface_api_key)
print("Google API Key:", google_api_key)
print("OpenAI API Key:", openai_api_key)
