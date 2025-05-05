# mi_agent/app_config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    output_dir: str = "data"
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0

    class Config:
        env_prefix = "MI_AGENT_"

# This is the singleton all modules will import:
settings = Settings()