from rag_benchmark.utils.config import (
    APP_ENV,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
)


def test_default_configuration():
    assert APP_ENV == "development"
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_CHUNK_SIZE == 500
    assert DEFAULT_CHUNK_OVERLAP == 100


def test_get_env_fallback_to_streamlit_secrets(monkeypatch):
    import sys
    from rag_benchmark.utils.config import get_env

    # When env var is not present, should check st.secrets
    monkeypatch.delenv("NON_EXISTENT_KEY_XYZ", raising=False)

    class DummyStreamlit:
        secrets = {"NON_EXISTENT_KEY_XYZ": "from_secrets"}

    monkeypatch.setitem(sys.modules, "streamlit", DummyStreamlit)

    val = get_env("NON_EXISTENT_KEY_XYZ")
    assert val == "from_secrets"