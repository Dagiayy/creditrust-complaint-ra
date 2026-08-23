from __future__ import annotations

import os

from creditrust.config import PROJECT_ROOT, Settings, get_settings


def test_default_paths_resolve_under_project_root():
    settings = Settings(_env_file=None)
    assert settings.vector_store_dir == PROJECT_ROOT / "vector_store" / "chroma_index"
    assert settings.filtered_data_path.is_relative_to(PROJECT_ROOT)


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("CREDITRUST_TOP_K", "9")
    monkeypatch.setenv("CREDITRUST_LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.top_k == 9
    assert settings.llm_provider == "mock"


def test_get_settings_is_cached():
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_relative_path_override_is_anchored_to_project_root(monkeypatch):
    monkeypatch.chdir(os.path.expanduser("~"))
    settings = Settings(_env_file=None, vector_store_dir="vector_store/chroma_index")
    assert settings.vector_store_dir == PROJECT_ROOT / "vector_store" / "chroma_index"
