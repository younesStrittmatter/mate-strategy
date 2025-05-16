import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "openai-key": os.getenv("OPENAI_KEY", "YOUR_OPENAI_KEY_HERE"),
    "assist-model": "gpt-4o-mini",
    "query-sleep": 2,
}
