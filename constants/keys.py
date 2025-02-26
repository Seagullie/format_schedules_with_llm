from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY_OR_NONE = os.getenv("GOOGLE_API_KEY")
OPEN_AI_API_KEY_OR_NONE = os.getenv("OPEN_AI_API_KEY")

# Validate that keys are present
if not GOOGLE_API_KEY_OR_NONE or not OPEN_AI_API_KEY_OR_NONE:
    raise ValueError("Missing required API keys in .env file")

GOOGLE_API_KEY = GOOGLE_API_KEY_OR_NONE
OPEN_AI_API_KEY = OPEN_AI_API_KEY_OR_NONE
