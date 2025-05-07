"""Global configuration and constants for InsightForge."""
import os, sys
# import getpass
from dotenv import load_dotenv, find_dotenv

# # 1) Try to load a .env automatically from cwd or any parent
# # 2) Allow override via MI_AGENT_ENV_FILE

# env_path = os.getenv("MI_AGENT_ENV_FILE") or find_dotenv()
# if env_path:
#     load_dotenv(env_path, override=False)

# def set_env(var: str):
#     """Prompt for and set an environment variable if not already set."""
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")

# 1) Manual check of CWD first
cwd_dotenv = os.path.join(os.getcwd(), ".env")
if os.path.isfile(cwd_dotenv):
    env_path = cwd_dotenv
else:
    # 2) Next, honor MI_AGENT_ENV_FILE
    env_path = os.getenv("MI_AGENT_ENV_FILE") or ""
    # 3) Finally, fallback to find_dotenv(usecwd=True)
    if not env_path:
        env_path = find_dotenv(usecwd=True) or ""

if env_path:
    load_dotenv(env_path, override=False)

# now prompt for anything still missing
def set_env(var: str):
    if var not in os.environ:
        import getpass
        os.environ[var] = getpass.getpass(f"{var}: ")

# load critical keys on import
set_env("OPENAI_API_KEY")
set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MI-Agent"
