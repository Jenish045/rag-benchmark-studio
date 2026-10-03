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