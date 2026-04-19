"""Tests for config loader with regex-based env var substitution."""
import os
import tempfile

import pytest

from inference.core.config import load_config


def test_load_config_basic():
    config = load_config("cyberverse_config.yaml")
    assert config["server"]["http_port"] == 8080
    assert config["inference"]["avatar"]["default"] in {"flash_head", "live_act"}
    assert set(config["inference"]["avatar"]["runtime"].keys()) == {
        "cuda_visible_devices",
        "world_size",
    }
    assert set(config["warmup"].keys()) == {"enabled", "distributed"}


def test_env_var_substitution():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("key: ${TEST_CYBERVERSE_VAR}\n")
        f.flush()
        os.environ["TEST_CYBERVERSE_VAR"] = "hello_world"
        try:
            config = load_config(f.name)
            assert config["key"] == "hello_world"
        finally:
            del os.environ["TEST_CYBERVERSE_VAR"]
            os.unlink(f.name)


def test_unmatched_env_var_preserved():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("key: ${NONEXISTENT_CYBERVERSE_VAR_12345}\n")
        f.flush()
        try:
            config = load_config(f.name)
            assert config["key"] == "${NONEXISTENT_CYBERVERSE_VAR_12345}"
        finally:
            os.unlink(f.name)


def test_path_not_substituted():
    """Ensure common env vars like PATH are NOT blindly substituted."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("my_path: /usr/local/bin\nother: literal_${not_a_var\n")
        f.flush()
        try:
            config = load_config(f.name)
            assert config["my_path"] == "/usr/local/bin"
        finally:
            os.unlink(f.name)


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_multiple_vars_in_one_line():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("url: ${TEST_HOST}:${TEST_PORT}\n")
        f.flush()
        os.environ["TEST_HOST"] = "localhost"
        os.environ["TEST_PORT"] = "8080"
        try:
            config = load_config(f.name)
            assert config["url"] == "localhost:8080"
        finally:
            del os.environ["TEST_HOST"]
            del os.environ["TEST_PORT"]
            os.unlink(f.name)
