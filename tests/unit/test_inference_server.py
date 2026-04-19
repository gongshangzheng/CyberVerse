from unittest.mock import MagicMock, patch

from inference.server import InferenceServer


def _make_server(config: dict) -> InferenceServer:
    with patch("inference.server.load_config", return_value=config):
        with patch("inference.server.grpc.aio.server", return_value=MagicMock()):
            return InferenceServer("cyberverse_config.yaml")


def test_build_plugin_config_passes_root_warmup_to_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.flash_head",
        {
            "plugin_class": "pkg.Plugin",
            "device": "cuda:0",
        },
    )

    assert plugin_config.plugin_name == "avatar.flash_head"
    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 2,
        "device": "cuda:0",
    }
    assert plugin_config.shared["warmup"] == config["warmup"]


def test_build_plugin_config_model_values_override_avatar_runtime_defaults():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.live_act",
        {
            "plugin_class": "pkg.Plugin",
            "world_size": 1,
        },
    )

    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 1,
    }


def test_build_plugin_config_does_not_pass_root_warmup_to_non_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "llm",
        "llm.openai",
        {
            "plugin_class": "pkg.Plugin",
            "model": "gpt-4o",
        },
    )

    assert plugin_config.plugin_name == "llm.openai"
    assert plugin_config.params == {"model": "gpt-4o"}
    assert plugin_config.shared == {}
