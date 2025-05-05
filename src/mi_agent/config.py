"""Global configuration and constants for InsightForge."""
import os
import getpass

# #: where all generated code & images & report go
# OUTPUT_DIR = os.getenv("MI_AGENT_OUTPUT_DIR", os.path.abspath("data"))

# #: which OpenAI model to use for all calls
# MODEL_NAME = os.getenv("MI_AGENT_MODEL_NAME", "gpt-4.1-mini")

# #: temperature or other defaults
# TEMPERATURE = float(os.getenv("MI_AGENT_TEMP", "0"))

def set_env(var: str):
    """Prompt for and set an environment variable if not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# load critical keys on import
set_env("OPENAI_API_KEY")
set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MI-Agent"
