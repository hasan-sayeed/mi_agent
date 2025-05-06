"""Global configuration and constants for InsightForge."""
import os
import getpass
from dotenv import load_dotenv, find_dotenv

# 1) Try to load a .env automatically from cwd or any parent
# 2) Allow override via MI_AGENT_ENV_FILE

env_path = os.getenv("MI_AGENT_ENV_FILE") or find_dotenv()
if env_path:
    load_dotenv(env_path, override=False)

def set_env(var: str):
    """Prompt for and set an environment variable if not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# load critical keys on import
set_env("OPENAI_API_KEY")
set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MI-Agent"
