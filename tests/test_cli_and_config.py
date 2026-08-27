"""CLI wiring and configuration precedence."""

from __future__ import annotations

import json

import pytest

from llmwiki import config
from llmwiki.cli import main
from llmwiki.errors import ProjectError
from llmwiki.project import open_project


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in list(dict(__import__("os").environ)):
        if name.startswith(("LLMWIKI_", "ANTHROPIC_", "OPENAI_")):
            monkeypatch.delenv(name, raising=False)
    # Keep the user's real config out of the tests.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_init_creates_the_three_layers(tmp_path, capsys):
    assert main(["init", str(tmp_path / "w"), "--template", "research", "--quiet"]) == 0
    root = tmp_path / "w"
    assert (root / "schema.md").exists()
    assert (root / "purpose.md").exists()
    assert (root / "raw" / "sources").is_dir()
    assert (root / "wiki" / "index.md").exists()
    assert "entity | wiki/entities/" in (root / "schema.md").read_text()


def test_init_refuses_to_clobber_an_existing_project(tmp_path):
    main(["init", str(tmp_path / "w"), "--quiet"])
    with pytest.raises(ProjectError, match="already contains"):
        from llmwiki.project import create

        create(tmp_path / "w")


def test_project_is_discovered_by_walking_up(tmp_path, monkeypatch):
    main(["init", str(tmp_path / "w"), "--quiet"])
    nested = tmp_path / "w" / "wiki" / "concepts"
    monkeypatch.chdir(nested)
    assert open_project().root == (tmp_path / "w").resolve()


def test_missing_project_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 1
    assert "no wiki project found" in capsys.readouterr().err


def test_status_json(tmp_path, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    assert main(["status", "-p", str(tmp_path / "w"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pages"] == 3  # index, log, overview
    assert payload["sources"] == 0
    assert payload["provider"] == "anthropic"


def test_search_command_runs_without_an_llm(tmp_path, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    project = open_project(tmp_path / "w")
    project.write(
        "wiki/concepts/storage.md",
        "---\ntype: concept\ntitle: Storage\n---\n\n# Storage\n\nGrid storage economics.\n",
    )
    assert main(["search", "grid", "storage", "-p", str(tmp_path / "w"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["path"] == "wiki/concepts/storage.md"


def test_ask_on_an_empty_project_does_not_need_credentials(tmp_path, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    assert main(["ask", "anything", "-p", str(tmp_path / "w"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["citations"] == []


def test_embed_without_configuration_is_a_clear_error(tmp_path, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    assert main(["embed", "-p", str(tmp_path / "w")]) == 1
    assert "LLMWIKI_EMBEDDING_MODEL" in capsys.readouterr().err


def test_config_precedence_env_over_file(tmp_path, monkeypatch):
    main(["init", str(tmp_path / "w"), "--quiet"])
    project = open_project(tmp_path / "w")
    project.save_settings({"llm": {"provider": "openai", "model": "from-file"}})

    from_file = config.load(project.root)
    assert from_file.llm.model == "from-file"

    monkeypatch.setenv("LLMWIKI_MODEL", "from-env")
    assert config.load(project.root).llm.model == "from-env"

    overridden = config.load(project.root, {"model": "from-flag"})
    assert overridden.llm.model == "from-flag"


def test_provider_detected_from_available_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert config.load().llm.provider == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert config.load().llm.provider == "anthropic"


def test_openai_provider_requires_an_explicit_model():
    settings = config.Settings(llm=config.LLMConfig(provider="openai"))
    with pytest.raises(config.ConfigError, match="no model configured"):
        settings.llm.resolved_model()


def test_anthropic_defaults_to_a_current_model():
    assert config.LLMConfig(provider="anthropic").resolved_model() == "claude-opus-5"


def test_add_reports_failure_with_a_nonzero_exit(tmp_path, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    unreadable = tmp_path / "thing.heic"
    unreadable.write_bytes(b"\x00\x01\x02")
    assert main(["add", str(unreadable), "-p", str(tmp_path / "w")]) == 1
    # An explicitly named file is always attempted, and says why it failed.
    assert "cannot read thing.heic" in capsys.readouterr().err


def test_recursive_add_skips_unsupported_files(tmp_path, monkeypatch):
    from llmwiki.cli import _expand

    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("a")
    (root / "sub" / "b.pdf").write_bytes(b"%PDF-")
    (root / "image.heic").write_bytes(b"\x00")
    (root / ".hidden.md").write_text("skip me")

    names = [p.name for p in _expand(root, recursive=True)]
    assert names == ["a.md", "b.pdf"]


def test_configured_but_unbuilt_vector_index_is_reported(tmp_path, monkeypatch, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    project = open_project(tmp_path / "w")
    project.write("wiki/concepts/a.md", "---\ntype: concept\ntitle: A\n---\n\n# A\n\nStorage.\n")
    monkeypatch.setenv("LLMWIKI_EMBEDDING_MODEL", "text-embedding-3-small")

    assert main(["search", "storage", "-p", str(tmp_path / "w")]) == 0
    assert "run 'llmwiki embed'" in capsys.readouterr().err


# ── strict project resolution ─────────────────────────────────────────────


def test_add_refuses_to_infer_the_project(tmp_path, monkeypatch, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    doc = tmp_path / "note.md"
    doc.write_text("# Note\n")
    monkeypatch.chdir(tmp_path / "w")  # inside the project — still not enough

    assert main(["add", str(doc)]) == 1
    assert "--project is required" in capsys.readouterr().err
    # Nothing was copied into raw/sources/ on the way to the error.
    assert list((tmp_path / "w" / "raw" / "sources").iterdir()) == []


def test_ask_refuses_to_infer_the_project(tmp_path, monkeypatch, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    monkeypatch.chdir(tmp_path / "w")
    assert main(["ask", "anything"]) == 1
    assert "--project is required" in capsys.readouterr().err


def test_an_exported_default_project_does_not_satisfy_strict_mode(tmp_path, monkeypatch, capsys):
    # The hazard strict mode exists for: an ambient default the caller never chose.
    main(["init", str(tmp_path / "w"), "--quiet"])
    monkeypatch.setenv("LLMWIKI_PROJECT", str(tmp_path / "w"))
    assert main(["ask", "anything"]) == 1
    assert "--project is required" in capsys.readouterr().err


def test_strict_mode_off_restores_discovery(tmp_path, monkeypatch, capsys):
    main(["init", str(tmp_path / "w"), "--quiet"])
    monkeypatch.setenv("LLMWIKI_STRICT_PROJECT", "0")
    monkeypatch.setenv("LLMWIKI_PROJECT", str(tmp_path / "w"))
    assert main(["ask", "anything", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["citations"] == []


def test_strict_mode_can_be_disabled_in_the_user_config(monkeypatch):
    path = config.user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"strict_project": False}))
    assert config.strict_project_enabled() is False

    monkeypatch.setenv("LLMWIKI_STRICT_PROJECT", "1")  # the environment still wins
    assert config.strict_project_enabled() is True


def test_read_only_commands_still_discover_the_project(tmp_path, monkeypatch):
    # status and search are how you work out where you are; requiring -p there
    # would leave the "no project" error with no way to act on it.
    main(["init", str(tmp_path / "w"), "--quiet"])
    monkeypatch.chdir(tmp_path / "w")
    assert main(["status", "--json"]) == 0
    assert main(["search", "storage", "--json"]) == 0


def test_ask_lists_the_cited_pages_and_summarizes_the_rest(tmp_path, capsys, monkeypatch):
    main(["init", str(tmp_path / "w"), "--quiet"])
    project = open_project(tmp_path / "w")
    for name in ("alpha", "beta", "gamma"):
        project.write(
            f"wiki/concepts/{name}.md",
            f"---\ntype: concept\ntitle: {name.title()}\n---\n\n# {name}\n\nStorage notes.\n",
        )

    from llmwiki.llm.base import Completion

    class _Stub:
        def complete(self, messages, *, max_tokens, on_token=None):
            return Completion(text="Storage, in short [2].", model="stub")

    monkeypatch.setattr("llmwiki.query.build_client", lambda _cfg: _Stub())
    assert main(["ask", "storage", "-p", str(tmp_path / "w")]) == 0

    err = capsys.readouterr().err
    assert "[2] wiki/concepts/beta.md" in err
    assert "alpha" not in err and "gamma" not in err
    assert "+2 retrieved pages the answer didn't cite" in err
    assert "3 packed · 1 cited" in err


def test_graph_annotations_are_capped(tmp_path):
    from llmwiki.cli import _via

    assert _via([]) == ""
    assert _via(["One", "Two"]) == "  (via One, Two)"
    assert _via(["A", "B", "C", "D", "E"]) == "  (via A, B, C +2 more)"
