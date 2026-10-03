import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{name}' must be an integer."
        ) from exc


APP_ENV = get_env("APP_ENV", "development")

LLM_API_KEY = get_env("LLM_API_KEY")

LLM_MODEL = get_env("LLM_MODEL")

DEFAULT_TOP_K = get_int_env("DEFAULT_TOP_K", 5)

DEFAULT_CHUNK_SIZE = get_int_env("DEFAULT_CHUNK_SIZE", 500)

DEFAULT_CHUNK_OVERLAP = get_int_env("DEFAULT_CHUNK_OVERLAP", 100)