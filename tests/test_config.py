"""Validate config constants are internally consistent."""
import src.config as config


class TestChunkSettings:
    def test_chunk_size_is_positive(self):
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_is_positive(self):
        assert config.CHUNK_OVERLAP > 0

    def test_overlap_smaller_than_chunk_size(self):
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_overlap_is_reasonable_fraction(self):
        ratio = config.CHUNK_OVERLAP / config.CHUNK_SIZE
        assert 0.05 <= ratio <= 0.5, f"Overlap ratio {ratio:.2f} is outside expected 5-50% range"


class TestLLMSettings:
    def test_temperature_is_zero_for_determinism(self):
        assert config.TEMPERATURE == 0.0

    def test_max_tokens_is_positive(self):
        assert config.MAX_TOKENS > 0

    def test_model_name_is_non_empty_string(self):
        assert isinstance(config.MODEL_NAME, str) and config.MODEL_NAME.strip()


class TestRetrievalSettings:
    def test_top_k_results_is_positive(self):
        assert config.TOP_K_RESULTS > 0

    def test_top_k_results_is_reasonable(self):
        assert 1 <= config.TOP_K_RESULTS <= 20


class TestMemorySettings:
    def test_memory_size_is_positive(self):
        assert config.MEMORY_SIZE > 0


class TestRetrySettings:
    def test_max_retries_at_least_one(self):
        assert config.MAX_RETRIES >= 1
