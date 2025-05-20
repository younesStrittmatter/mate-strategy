import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "openai-key": os.getenv("OPENAI_KEY", "YOUR_OPENAI_KEY_HERE"),
    "assist-model": os.getenv("ASSIST_MODEL", "gpt-4o-mini"),
    "query-sleep": os.getenv("QUERY_SLEEP", "0.5"),
}
