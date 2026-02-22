import pytest
from types import SimpleNamespace

from evalview.skills.runner import SkillRunner


_PROVIDER_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "KIMI_BASE_URL",
    "MOONSHOT_BASE_URL",
    "SKILL_TEST_PROVIDER",
    "SKILL_TEST_API_KEY",
    "SKILL_TEST_BASE_URL",
]


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_openai_alias_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY detected but no base URL configured"):
        SkillRunner(model="deepseek-chat")


def test_openai_alias_with_base_url_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    runner = SkillRunner(model="deepseek-chat")

    assert runner.provider == "openai"
    assert runner.base_url == "https://api.deepseek.com/v1"


def test_openai_key_without_base_url_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    runner = SkillRunner(model="gpt-4o-mini")

    assert runner.provider == "openai"
    assert runner.base_url is None


def test_openai_key_ignores_other_provider_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    runner = SkillRunner(model="gpt-4o-mini")

    assert runner.provider == "openai"
    assert runner.base_url is None


def test_alias_key_uses_matching_base_url_not_other_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    runner = SkillRunner(model="deepseek-chat")

    assert runner.provider == "openai"
    assert runner.base_url == "https://api.deepseek.com/v1"


def test_skill_test_base_url_overrides_alias_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("SKILL_TEST_BASE_URL", "https://proxy.internal/v1")

    runner = SkillRunner(model="deepseek-chat")

    assert runner.provider == "openai"
    assert runner.base_url == "https://proxy.internal/v1"


def test_invoke_model_normalizes_anthropic_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    runner = SkillRunner(model="claude-test")
    runner._client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[SimpleNamespace(text="Hello"), SimpleNamespace(text="World")],
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )
        )
    )

    output, input_tokens, output_tokens = runner._invoke_model(
        model="claude-test",
        system_prompt="sys",
        user_input="hi",
    )

    assert output == "Hello\nWorld"
    assert input_tokens == 11
    assert output_tokens == 7


def test_invoke_model_normalizes_openai_compatible_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SKILL_TEST_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    runner = SkillRunner(model="gpt-test")
    runner._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=[{"type": "text", "text": "Alpha"}, {"type": "text", "text": "Beta"}]
                            )
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=13, completion_tokens=9),
                )
            )
        )
    )

    output, input_tokens, output_tokens = runner._invoke_model(
        model="gpt-test",
        system_prompt="sys",
        user_input="hi",
    )

    assert output == "Alpha\nBeta"
    assert input_tokens == 13
    assert output_tokens == 9


@pytest.mark.parametrize(
    ("message", "expected_prefix"),
    [
        ("Invalid API key", "Authentication error:"),
        ("Connection error.", "Connection error:"),
        ("Request timeout after 30s", "Timeout error:"),
        ("DEEPSEEK_API_KEY detected but no base URL configured", "Provider configuration error:"),
    ],
)
def test_error_categorization_labels(message: str, expected_prefix: str) -> None:
    categorized = SkillRunner._categorize_model_error(RuntimeError(message))
    assert categorized.startswith(expected_prefix)
