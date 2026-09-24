"""Conditional compression routing via ``auxiliary.compression.local_override``.

When the live main model runs on one of the listed LAN base_urls, compression
swaps its routing fields to the override block (e.g. an OpenRouter preset).
When the main model is anywhere else, compression resolves exactly as before.
The main runtime is read per call, so a model switch cannot leak a stale route.
"""

import agent.auxiliary_client as aux

_OVERRIDE = {
    "provider": "openrouter",
    "model": "@preset/local-compression",
    "base_urls": ["http://192.168.8.179:18080/v1", "http://myllm:18080"],
}


def _patch_task_config(monkeypatch, compression_cfg):
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: dict(compression_cfg) if task == "compression" else {},
    )


def _resolve(monkeypatch, compression_cfg, main_runtime):
    _patch_task_config(monkeypatch, compression_cfg)
    return aux._resolve_task_provider_model("compression", main_runtime=main_runtime)


def test_local_main_model_uses_override_route(monkeypatch):
    cfg = {"provider": "", "model": "", "local_override": _OVERRIDE}
    provider, model, base_url, _key, _mode = _resolve(
        monkeypatch, cfg, {"provider": "custom", "base_url": "http://192.168.8.179:18080/v1"}
    )
    assert provider == "openrouter"
    assert model == "@preset/local-compression"


def test_trailing_slash_and_host_form_both_match(monkeypatch):
    cfg = {"provider": "", "model": "", "local_override": _OVERRIDE}
    provider, _model, *_rest = _resolve(
        monkeypatch, cfg, {"provider": "custom", "base_url": "http://192.168.8.179:18080/v1/"}
    )
    assert provider == "openrouter"
    provider, _model, *_rest = _resolve(
        monkeypatch, cfg, {"provider": "custom", "base_url": "http://myllm:18080"}
    )
    assert provider == "openrouter"


def test_external_main_model_keeps_plain_compression_route(monkeypatch):
    cfg = {"provider": "", "model": "", "local_override": _OVERRIDE}
    provider, model, *_rest = _resolve(
        monkeypatch, cfg, {"provider": "alibaba-token-plan", "base_url": "https://token-plan.example/v1"}
    )
    assert provider == "auto"
    assert model is None


def test_no_override_block_is_unchanged_behaviour(monkeypatch):
    cfg = {"provider": "deepseek", "model": "deepseek-v4-flash"}
    provider, model, *_rest = _resolve(
        monkeypatch, cfg, {"provider": "custom", "base_url": "http://myllm:18080"}
    )
    assert provider == "deepseek"
    assert model == "deepseek-v4-flash"


def test_explicit_override_beats_local_override(monkeypatch):
    """An explicit provider arg (caller-forced route) still wins over the block."""
    cfg = {"provider": "", "model": "", "local_override": _OVERRIDE}
    _patch_task_config(monkeypatch, cfg)
    provider, _model, *_rest = aux._resolve_task_provider_model(
        "compression", provider="anthropic",
        main_runtime={"provider": "custom", "base_url": "http://myllm:18080"},
    )
    assert provider == "anthropic"


def test_malformed_override_is_ignored(monkeypatch):
    for bad in ({"local_override": "not-a-dict"},
                {"local_override": {"provider": "openrouter", "base_urls": "http://myllm:18080"}},
                {"local_override": {"provider": "openrouter", "base_urls": []}}):
        cfg = dict(bad); cfg.setdefault("provider", ""); cfg.setdefault("model", "")
        provider, _model, *_rest = _resolve(
            monkeypatch, cfg, {"provider": "custom", "base_url": "http://myllm:18080"}
        )
        assert provider == "auto"


def test_runtime_without_base_url_does_not_match(monkeypatch):
    cfg = {"provider": "", "model": "", "local_override": _OVERRIDE}
    provider, *_rest = _resolve(monkeypatch, cfg, {"provider": "custom", "base_url": ""})
    assert provider == "auto"


def test_client_getters_thread_main_runtime(monkeypatch):
    seen = {}

    def _fake_resolve(task=None, provider=None, model=None, base_url=None, api_key=None, *, main_runtime=None):
        seen["main_runtime"] = main_runtime
        return "auto", None, None, None, None

    monkeypatch.setattr(aux, "_resolve_task_provider_model", _fake_resolve)
    monkeypatch.setattr(aux, "resolve_provider_client", lambda *a, **k: (None, None))
    runtime = {"provider": "custom", "base_url": "http://myllm:18080", "model": "Huihui"}
    aux.get_text_auxiliary_client("compression", main_runtime=runtime)
    assert seen["main_runtime"] is runtime
