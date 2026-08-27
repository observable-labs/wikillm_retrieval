from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmwiki import config
from llmwiki.dotenv import dotenv_candidates, load_dotenv, parse_dotenv


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """No ambient LLMWIKI_* or stray `.env` on the walk-up path.

    Conftest stubs out `config.load_dotenv` for the rest of the suite; this
    is the module that needs the real one.
    """
    monkeypatch.setattr(config, "load_dotenv", load_dotenv)
    for key in list(os.environ):
        if key.startswith(("LLMWIKI_", "DOTENV_TEST_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path / "home")


def test_parses_the_shapes_a_shell_accepts():
    values, warnings = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=one",
                "export EXPORTED=two",
                "SPACED = three ",
                'DOUBLE="four"',
                "SINGLE='five'",
                "EMPTY=",
                "TRAILING=six  # not part of the value",
                "HASH_IN_VALUE=a#b",
                'URL="https://example.test/v1beta/openai/chat/completions"',
            ]
        )
    )
    assert warnings == []
    assert values == {
        "PLAIN": "one",
        "EXPORTED": "two",
        "SPACED": "three",
        "DOUBLE": "four",
        "SINGLE": "five",
        "EMPTY": "",
        "TRAILING": "six",
        "HASH_IN_VALUE": "a#b",
        "URL": "https://example.test/v1beta/openai/chat/completions",
    }


def test_expands_references_to_earlier_lines_and_the_environment(monkeypatch):
    """The shape that actually ships in env-template.txt:

        GEMINI_API_KEY=secret
        LLMWIKI_API_KEY="$GEMINI_API_KEY"

    Sourcing the file under `set -a` expands this; a naive parser would set
    the literal string `$GEMINI_API_KEY` and the request would 401 with a
    key that looks superficially plausible in a config dump.
    """
    monkeypatch.setenv("DOTENV_TEST_OUTER", "from-env")
    values, warnings = parse_dotenv(
        "\n".join(
            [
                "KEY=secret",
                'DERIVED="$KEY"',
                "BRACED=${KEY}-suffix",
                'MIXED="$DOTENV_TEST_OUTER/$KEY"',
                "UNSET=[$DOTENV_TEST_MISSING]",
                r'ESCAPED="\$KEY"',
                r"LITERAL='$KEY'",
            ]
        )
    )
    assert warnings == []
    assert values["DERIVED"] == "secret"
    assert values["BRACED"] == "secret-suffix"
    assert values["MIXED"] == "from-env/secret"
    assert values["UNSET"] == "[]"
    assert values["ESCAPED"] == "$KEY"
    assert values["LITERAL"] == "$KEY"


def test_double_quoted_escapes_and_single_quoted_literals():
    values, warnings = parse_dotenv(
        "\n".join(
            [
                r'NEWLINE="a\nb"',
                r"RAW='a\nb'",
                r'QUOTED="say \"hi\""',
            ]
        )
    )
    assert warnings == []
    assert values["NEWLINE"] == "a\nb"
    assert values["RAW"] == r"a\nb"
    assert values["QUOTED"] == 'say "hi"'


def test_unparseable_lines_are_reported_rather_than_dropped():
    values, warnings = parse_dotenv(
        "\n".join(
            [
                "GOOD=yes",
                "this is not an assignment",
                'OPEN="unterminated',
                "ALSO_GOOD=yes",
            ]
        )
    )
    assert values == {"GOOD": "yes", "ALSO_GOOD": "yes"}
    assert len(warnings) == 2
    assert "line 2" in warnings[0]
    assert "line 3" in warnings[1] and "unterminated" in warnings[1]


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLMWIKI_MODEL=from-file\nLLMWIKI_PROVIDER=openai\n")
    monkeypatch.setenv("LLMWIKI_MODEL", "from-shell")

    result = load_dotenv()

    assert os.environ["LLMWIKI_MODEL"] == "from-shell"
    assert os.environ["LLMWIKI_PROVIDER"] == "openai"
    assert "LLMWIKI_MODEL" in result.skipped
    assert "LLMWIKI_PROVIDER" in result.applied


def test_project_env_outranks_the_working_directory(tmp_path, monkeypatch):
    project = tmp_path / "wiki"
    project.mkdir()
    (project / ".env").write_text("LLMWIKI_MODEL=project\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("LLMWIKI_MODEL=cwd\nLLMWIKI_EFFORT=high\n")
    monkeypatch.chdir(elsewhere)

    load_dotenv(project)

    assert os.environ["LLMWIKI_MODEL"] == "project"
    # The lower-precedence file still contributes names the winner didn't set.
    assert os.environ["LLMWIKI_EFFORT"] == "high"


def test_walks_up_from_the_working_directory_but_stops_below_home(tmp_path, monkeypatch):
    home = Path(os.environ["HOME"])
    repo = home / "repo"
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)
    (repo / ".env").write_text("LLMWIKI_MODEL=from-repo-root\n")
    (home / ".env").write_text("LLMWIKI_PROVIDER=should-not-load\n")
    monkeypatch.chdir(nested)

    result = load_dotenv()

    assert os.environ["LLMWIKI_MODEL"] == "from-repo-root"
    assert "LLMWIKI_PROVIDER" not in os.environ
    assert home / ".env" not in dotenv_candidates()
    assert result.paths == [repo / ".env"]


def test_explicit_path_and_the_off_switch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLMWIKI_MODEL=ignored\n")
    (tmp_path / "custom.env").write_text("LLMWIKI_MODEL=explicit\n")

    monkeypatch.setenv("LLMWIKI_DOTENV", str(tmp_path / "custom.env"))
    load_dotenv()
    assert os.environ["LLMWIKI_MODEL"] == "explicit"

    monkeypatch.delenv("LLMWIKI_MODEL")
    monkeypatch.setenv("LLMWIKI_DOTENV", "0")
    result = load_dotenv()
    assert "LLMWIKI_MODEL" not in os.environ
    assert result.paths == []


def test_settings_pick_up_the_provider_from_a_project_env(tmp_path, monkeypatch):
    """The regression this exists for: with nothing exported, provider
    detection falls through to Anthropic and the run dies on the wrong
    credential while the intended key sits unread in `.env`."""
    project = tmp_path / "wiki"
    (project / ".llm-wiki").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    assert config.load(project).llm.provider == "anthropic"

    (project / ".env").write_text(
        "GEMINI_API_KEY=test-key\n"
        "LLMWIKI_PROVIDER=openai\n"
        'LLMWIKI_API_KEY="$GEMINI_API_KEY"\n'
        "LLMWIKI_MODEL=gemini-3.7-flash\n"
        "LLMWIKI_BASE_URL=https://example.test/v1beta/openai/chat/completions\n"
    )

    settings = config.load(project)
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gemini-3.7-flash"
    assert settings.llm.api_key == "test-key"
    assert settings.llm.base_url.endswith("/chat/completions")


def test_cli_flags_still_outrank_the_file(tmp_path, monkeypatch):
    project = tmp_path / "wiki"
    (project / ".llm-wiki").mkdir(parents=True)
    (project / ".env").write_text("LLMWIKI_MODEL=from-file\nLLMWIKI_PROVIDER=openai\n")
    monkeypatch.chdir(tmp_path)

    settings = config.load(project, {"model": "from-flag"})
    assert settings.llm.model == "from-flag"
