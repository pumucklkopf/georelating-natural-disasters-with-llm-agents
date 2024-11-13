from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
api_key = os.getenv('SAIA_API_KEY')

