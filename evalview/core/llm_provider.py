"""Multi-provider LLM client for LLM-as-judge evaluation.

Supports OpenAI, Anthropic, Gemini, and Grok with automatic provider detection.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers for evaluation."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    env_var: str
    default_model: str
    display_name: str
    api_key_url: str


# Provider configurations
PROVIDER_CONFIGS: Dict[LLMProvider, ProviderConfig] = {
    LLMProvider.OPENAI: ProviderConfig(
        name="openai",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        display_name="OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
    ),
    LLMProvider.ANTHROPIC: ProviderConfig(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-5-20250929",
        display_name="Anthropic",
        api_key_url="https://console.anthropic.com/settings/keys",
    ),
    LLMProvider.GEMINI: ProviderConfig(
        name="gemini",
        env_var="GEMINI_API_KEY",
        default_model="gemini-2.0-flash",
        display_name="Google Gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
    ),
    LLMProvider.GROK: ProviderConfig(
        name="grok",
        env_var="XAI_API_KEY",
        default_model="grok-2-latest",
        display_name="xAI Grok",
        api_key_url="https://console.x.ai/",
    ),
    LLMProvider.HUGGINGFACE: ProviderConfig(
        name="huggingface",
        env_var="HF_TOKEN",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Hugging Face",
        api_key_url="https://huggingface.co/settings/tokens",
    ),
    LLMProvider.OLLAMA: ProviderConfig(
        name="ollama",
        env_var="OLLAMA_HOST",  # Optional - defaults to localhost:11434
        default_model="llama3.2",
        display_name="Ollama (Local)",
        api_key_url="https://ollama.ai/download",  # Download page, no API key needed
    ),
}

# Model aliases for better DX - shortcuts map to full model names
MODEL_ALIASES: Dict[str, str] = {
    # OpenAI GPT-5 family (use simple names - they track latest)
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5.1": "gpt-5.1",
    # OpenAI GPT-4 family
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4": "gpt-4-turbo",
    # Anthropic Claude
    "sonnet": "claude-sonnet-4-5-20250929",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-5-20251101",
    "claude-opus": "claude-opus-4-5-20251101",
    "haiku": "claude-3-5-haiku-latest",
    "claude-haiku": "claude-3-5-haiku-latest",
    # HuggingFace Llama
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    # Google Gemini
    "gemini": "gemini-3.0",
    "gemini-3": "gemini-3.0",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
    # Ollama (local models)
    "ollama-llama": "llama3.2",
    "llama3.2": "llama3.2",
    "llama3.1": "llama3.1",
    "mistral": "mistral",
    "codellama": "codellama",
    "phi": "phi",
}


def resolve_model_alias(model: str) -> str:
    """Resolve model alias to full model name.

    Args:
        model: Model name or alias (e.g., 'gpt-5', 'sonnet', 'llama-70b')

    Returns:
        Full model name (e.g., 'gpt-5-2025-08-07', 'claude-sonnet-4-5-20250929')
    """
    return MODEL_ALIASES.get(model.lower(), model)


def is_ollama_running() -> bool:
    """Check if Ollama is running locally.

    Returns:
        True if Ollama is accessible at localhost:11434
    """
    import socket

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Parse host and port from URL
    host = ollama_host.replace("http://", "").replace("https://", "")
    if ":" in host:
        host, port_str = host.split(":", 1)
        port = int(port_str)
    else:
        port = 11434

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            return True
    except (socket.timeout, socket.error, OSError):
        return False


def detect_available_providers() -> List[Tuple[LLMProvider, str]]:
    """Detect which LLM providers have API keys configured.

    For most providers, checks if the environment variable is set.
    For Ollama, checks if the server is running locally (no API key needed).

    Returns:
        List of (provider, api_key) tuples for available providers.
    """
    available: List[Tuple[LLMProvider, str]] = []
    for provider, config in PROVIDER_CONFIGS.items():
        if provider == LLMProvider.OLLAMA:
            # Ollama doesn't need an API key - check if it's running
            if is_ollama_running():
                available.append((provider, "ollama"))  # Placeholder "key"
        else:
            api_key = os.getenv(config.env_var)
            if api_key:
                available.append((provider, api_key))
    return available


def get_provider_from_env() -> Optional[LLMProvider]:
    """Get the user-selected provider from EVAL_PROVIDER env var."""
    provider_name = os.getenv("EVAL_PROVIDER", "").lower()
    if not provider_name:
        return None

    for provider in LLMProvider:
        if provider.value == provider_name:
            return provider
    return None


def select_provider() -> Tuple[LLMProvider, str]:
    """Select the best available LLM provider.

    Priority:
    1. User-specified EVAL_PROVIDER environment variable
    2. First available provider in order: OpenAI, Anthropic, Gemini, Grok

    Returns:
        Tuple of (provider, api_key)

    Raises:
        ValueError: If no provider is available
    """
    available = detect_available_providers()

    if not available:
        raise ValueError("No LLM provider API key found")

    # Check if user specified a provider
    user_provider = get_provider_from_env()
    if user_provider:
        # Special case for Ollama - check if running, not env var
        if user_provider == LLMProvider.OLLAMA:
            if is_ollama_running():
                return user_provider, "ollama"
            else:
                raise ValueError(
                    "EVAL_PROVIDER=ollama specified but Ollama is not running. Start with: ollama serve"
                )

        for provider, api_key in available:
            if provider == user_provider:
                return provider, api_key
        # User specified a provider but it's not available
        config = PROVIDER_CONFIGS[user_provider]
        raise ValueError(
            f"EVAL_PROVIDER={user_provider.value} specified but {config.env_var} not set"
        )

    # Return first available provider
    return available[0]


class JudgeCostTracker:
    """Track LLM-as-judge API costs across all evaluations."""

    # Pricing per 1M tokens (input, output)
    PRICING = {
        "openai": {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.00, 30.00),
        },
        "anthropic": {
            "claude-sonnet-4-5-20250929": (3.00, 15.00),
            "claude-3-5-haiku-latest": (0.25, 1.25),
            "claude-opus-4-5-20251101": (15.00, 75.00),
        },
        "gemini": {
            "gemini-2.0-flash": (0.10, 0.40),
            "gemini-1.5-pro": (1.25, 5.00),
        },
        "ollama": {},  # Free - local
        "huggingface": {
            "meta-llama/Llama-3.1-8B-Instruct": (0.05, 0.05),
        },
    }

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    def add_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Track token usage and calculate cost."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        # Calculate cost
        pricing = self.PRICING.get(provider, {})
        model_pricing = None

        # Try exact match first, then partial match
        for model_name, prices in pricing.items():
            if model_name in model or model in model_name:
                model_pricing = prices
                break

        if model_pricing:
            input_cost = (input_tokens / 1_000_000) * model_pricing[0]
            output_cost = (output_tokens / 1_000_000) * model_pricing[1]
            self.total_cost += input_cost + output_cost

    def get_summary(self) -> str:
        """Get a summary string of costs."""
        if self.call_count == 0:
            return "No judge calls yet"

        total_tokens = self.total_input_tokens + self.total_output_tokens

        if self.total_cost > 0:
            # Paid API - show cost prominently
            return f"${self.total_cost:.4f} | {total_tokens:,} tokens ({self.call_count} calls)"
        else:
            # Free (Ollama) - just show tokens
            return f"FREE | {total_tokens:,} tokens ({self.call_count} calls)"

    def get_detailed_summary(self) -> str:
        """Get detailed breakdown of costs."""
        if self.call_count == 0:
            return "No judge calls yet"

        lines = []
        lines.append(f"Judge LLM Usage:")
        lines.append(f"  Calls:         {self.call_count}")
        lines.append(f"  Input tokens:  {self.total_input_tokens:,}")
        lines.append(f"  Output tokens: {self.total_output_tokens:,}")
        lines.append(f"  Total tokens:  {self.total_input_tokens + self.total_output_tokens:,}")

        if self.total_cost > 0:
            lines.append(f"  Total cost:    ${self.total_cost:.4f}")
        else:
            lines.append(f"  Total cost:    FREE (local model)")

        return "\n".join(lines)

    def reset(self):
        """Reset all counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0


# Global cost tracker instance
judge_cost_tracker = JudgeCostTracker()


class LLMClient:
    """Multi-provider LLM client for evaluation."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            provider: LLM provider to use (auto-detected if not specified)
            api_key: API key (uses env var if not specified)
            model: Model to use (uses provider default if not specified)
        """
        if provider and api_key:
            self.provider = provider
            self.api_key = api_key
        else:
            self.provider, self.api_key = select_provider()

        self.config = PROVIDER_CONFIGS[self.provider]

        # Allow EVAL_MODEL to override the default model
        self.model = model or os.getenv("EVAL_MODEL") or self.config.default_model

        logger.info(f"Using {self.config.display_name} ({self.model}) for LLM-as-judge")

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Make a chat completion request and return parsed JSON.

        Args:
            system_prompt: System message
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON response from the LLM
        """
        if self.provider == LLMProvider.OPENAI:
            return await self._openai_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.GEMINI:
            return await self._gemini_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.GROK:
            return await self._grok_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.HUGGINGFACE:
            return await self._huggingface_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        elif self.provider == LLMProvider.OLLAMA:
            return await self._ollama_completion(
                system_prompt, user_prompt, temperature, max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _openai_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """OpenAI chat completion."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)

        # GPT-5 models require temperature=1 and max_completion_tokens
        is_gpt5 = self.model.startswith("gpt-5")

        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 1 if is_gpt5 else temperature,
        }

        if is_gpt5:
            params["max_completion_tokens"] = max_tokens * 5
        else:
            params["max_tokens"] = max_tokens

        response = await client.chat.completions.create(**params)  # type: ignore[call-overload]

        # Track usage
        if response.usage:
            judge_cost_tracker.add_usage(
                "openai", self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

        return json.loads(response.choices[0].message.content or "{}")

    async def _anthropic_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Anthropic chat completion."""
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)

        # Anthropic requires explicit JSON instruction in prompt
        json_instruction = "\n\nRespond with ONLY a valid JSON object, no other text."

        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt + json_instruction,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )

        # Track usage
        if response.usage:
            judge_cost_tracker.add_usage(
                "anthropic", self.model,
                response.usage.input_tokens,
                response.usage.output_tokens
            )

        # Extract text from response
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Parse JSON from response (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())

    async def _gemini_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Google Gemini chat completion."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Google GenAI package required. Install with: pip install google-genai"
            )

        client = genai.Client(api_key=self.api_key)

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )

        return json.loads(response.text or "{}")

    async def _grok_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """xAI Grok chat completion (OpenAI-compatible API)."""
        from openai import AsyncOpenAI

        # Grok uses OpenAI-compatible API
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return json.loads(response.choices[0].message.content or "{}")

    async def _huggingface_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Hugging Face Inference API chat completion (OpenAI-compatible)."""
        from openai import AsyncOpenAI

        # HF Inference Providers - unified router endpoint (2025)
        # Routes to best available provider (Together, Fireworks, etc.)
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://router.huggingface.co/v1",
        )

        # Add explicit JSON instruction since not all models support response_format
        json_instruction = "\n\nRespond with ONLY a valid JSON object, no other text."

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt + json_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse JSON from response (handle markdown code blocks)
        text = response.choices[0].message.content or "{}"
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain extra content.

        Local LLMs sometimes add explanations before/after JSON.
        This tries multiple strategies to find valid JSON.
        """
        import re

        text = text.strip()

        # Strategy 1: Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks
        if "```" in text:
            # Extract content between code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find JSON object pattern { ... }
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Return a default evaluation if we can't parse
        # Look for keywords to make a best-effort score
        logger.warning(f"Could not parse JSON from Ollama response: {text[:200]}...")

        # Try to extract a score from the text if mentioned
        score_match = re.search(r'(\d{1,3})(?:/100|%| out of 100| points)', text.lower())
        score = int(score_match.group(1)) if score_match else 70

        return {
            "score": min(score, 100),
            "reasoning": f"Auto-extracted from non-JSON response: {text[:500]}",
            "strengths": ["Response generated"],
            "weaknesses": ["Model did not return valid JSON format"]
        }

    async def _ollama_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Ollama local LLM chat completion (OpenAI-compatible)."""
        from openai import AsyncOpenAI

        # Ollama runs locally with OpenAI-compatible API
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        client = AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't need an API key
            base_url=f"{ollama_host}/v1",
        )

        # Add explicit JSON instruction with example format
        json_instruction = """

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{"score": <number 0-100>, "reasoning": "<string>", "strengths": ["<string>"], "weaknesses": ["<string>"]}

Do not include any text before or after the JSON. Do not use markdown code blocks."""

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt + json_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Track usage (Ollama is free but we track tokens for visibility)
        if response.usage:
            judge_cost_tracker.add_usage(
                "ollama", self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

        text = response.choices[0].message.content or "{}"
        return self._extract_json_from_text(text)


def get_missing_provider_message() -> str:
    """Generate a helpful error message when no provider is available."""
    lines = [
        "\n[red bold]No LLM provider API key found[/red bold]\n",
        "EvalView uses LLM-as-judge to evaluate output quality.",
        "Please set at least one of these API keys:\n",
    ]

    for provider, config in PROVIDER_CONFIGS.items():
        lines.append(f"  [cyan]{config.env_var}[/cyan] - {config.display_name}")

    lines.append("\nExample:")
    lines.append("  [cyan]export ANTHROPIC_API_KEY='your-key-here'[/cyan]")
    lines.append("\nOr add to your .env file:")
    lines.append("  [cyan]echo 'ANTHROPIC_API_KEY=your-key-here' >> .env[/cyan]")
    lines.append("\n[dim]Tip: Set EVAL_PROVIDER to choose a specific provider (openai, anthropic, gemini, grok, huggingface, ollama)[/dim]")
    lines.append("[dim]Tip: Set EVAL_MODEL to use a specific model[/dim]")
    lines.append("[dim]Tip: Use Ollama for free local evaluation - just run 'ollama serve' (no API key needed)[/dim]\n")
    lines.append("Get API keys at:")

    for provider, config in PROVIDER_CONFIGS.items():
        lines.append(f"  • {config.display_name}: {config.api_key_url}")

    return "\n".join(lines)


def get_provider_status() -> str:
    """Get a status message showing available providers."""
    available = detect_available_providers()

    if not available:
        return get_missing_provider_message()

    lines = ["[dim]Available LLM providers:[/dim]"]
    for provider, _ in available:
        config = PROVIDER_CONFIGS[provider]
        lines.append(f"  [green]✓[/green] {config.display_name} ({config.env_var})")

    # Show which one will be used
    try:
        selected, _ = select_provider()
        config = PROVIDER_CONFIGS[selected]
        model = os.getenv("EVAL_MODEL") or config.default_model
        lines.append(f"\n[dim]Using: {config.display_name} ({model})[/dim]")
    except ValueError:
        pass

    return "\n".join(lines)


def interactive_provider_selection(console) -> Optional[Tuple[LLMProvider, str]]:
    """Interactively ask user which provider to use.

    Args:
        console: Rich console for output

    Returns:
        Tuple of (provider, api_key) or None if user needs to add a key
    """
    available = detect_available_providers()
    available_providers = {p for p, _ in available}

    console.print("\n[bold]Select LLM provider for evaluation:[/bold]\n")

    # Build choices list
    choices = []
    for i, (provider, config) in enumerate(PROVIDER_CONFIGS.items(), 1):
        has_key = provider in available_providers
        if has_key:
            choices.append((provider, config, True))
            console.print(f"  [green]{i}. {config.display_name}[/green] [dim](API key found)[/dim]")
        else:
            choices.append((provider, config, False))
            console.print(f"  [dim]{i}. {config.display_name}[/dim]")

    # Show recommendation if any keys are available
    if available:
        available_names = [PROVIDER_CONFIGS[p].display_name for p, _ in available]
        console.print(f"\n[dim]Recommended: {', '.join(available_names)} (API key already set)[/dim]")

    console.print()

    # Get user choice
    while True:
        try:
            choice = console.input("[bold]Enter choice (1-5): [/bold]").strip()
            if not choice:
                # Default to first available provider
                if available:
                    provider, api_key = available[0]
                    config = PROVIDER_CONFIGS[provider]
                    console.print(f"\n[green]Using {config.display_name}[/green]")
                    return provider, api_key
                else:
                    console.print("[yellow]No provider selected.[/yellow]")
                    return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(choices):
                provider, config, has_key = choices[choice_idx]

                if has_key:
                    # User chose a provider they have a key for
                    api_key = os.getenv(config.env_var) or ""
                    console.print(f"\n[green]Using {config.display_name}[/green]")
                    return provider, api_key
                else:
                    # User chose a provider without a key - offer alternatives
                    console.print(f"\n[yellow]You don't have an API key for {config.display_name}.[/yellow]")

                    if available:
                        # Offer to use an available provider instead
                        available_names = [PROVIDER_CONFIGS[p].display_name for p, _ in available]
                        console.print("\n[bold]What would you like to do?[/bold]")
                        console.print(f"  [cyan]1.[/cyan] Add {config.display_name} API key")
                        for i, (avail_provider, _) in enumerate(available, 2):
                            avail_config = PROVIDER_CONFIGS[avail_provider]
                            console.print(f"  [green]{i}.[/green] Use {avail_config.display_name} instead [dim](API key available)[/dim]")

                        console.print()
                        sub_choice = console.input("[bold]Enter choice: [/bold]").strip()

                        if sub_choice == "1":
                            # User wants to add the key
                            console.print(f"\n[bold]To add {config.display_name} API key:[/bold]")
                            console.print(f"  [cyan]export {config.env_var}='your-key-here'[/cyan]")
                            console.print("\nOr add to .env.local:")
                            console.print(f"  [cyan]echo '{config.env_var}=your-key-here' >> .env.local[/cyan]")
                            console.print(f"\n[dim]Get your API key at: {config.api_key_url}[/dim]")
                            return None
                        else:
                            # User wants to use an available provider
                            try:
                                sub_idx = int(sub_choice) - 2
                                if 0 <= sub_idx < len(available):
                                    alt_provider, alt_api_key = available[sub_idx]
                                    alt_config = PROVIDER_CONFIGS[alt_provider]
                                    console.print(f"\n[green]Using {alt_config.display_name}[/green]")
                                    return alt_provider, alt_api_key
                            except (ValueError, IndexError):
                                pass
                            console.print("[red]Invalid choice.[/red]")
                            # Loop back to main selection
                            continue
                    else:
                        # No alternatives available
                        console.print(f"\n[bold]To add {config.display_name} API key:[/bold]")
                        console.print(f"  [cyan]export {config.env_var}='your-key-here'[/cyan]")
                        console.print("\nOr add to .env.local:")
                        console.print(f"  [cyan]echo '{config.env_var}=your-key-here' >> .env.local[/cyan]")
                        console.print(f"\n[dim]Get your API key at: {config.api_key_url}[/dim]")
                        return None
            else:
                console.print("[red]Invalid choice. Please enter 1-5.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number 1-5.[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            return None


def save_provider_preference(provider: LLMProvider) -> None:
    """Save provider preference to .env.local file."""
    env_file = ".env.local"
    provider_line = f"EVAL_PROVIDER={provider.value}\n"

    # Read existing content
    existing_lines = []
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            existing_lines = f.readlines()

    # Remove existing EVAL_PROVIDER line if present
    new_lines = [line for line in existing_lines if not line.startswith("EVAL_PROVIDER=")]

    # Ensure last line ends with newline before appending
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    # Add new provider preference
    new_lines.append(provider_line)

    # Write back
    with open(env_file, "w") as f:
        f.writelines(new_lines)


def get_or_select_provider(console, force_interactive: bool = False) -> Optional[Tuple[LLMProvider, str]]:
    """Get provider from env or interactively select one.

    Logic:
    1. If EVAL_PROVIDER is set and has API key -> use it (no prompt)
    2. If only one provider has a key -> use it automatically (no prompt)
    3. If multiple providers have keys -> ask user to choose
    4. If no providers have keys -> show error

    Args:
        console: Rich console for output
        force_interactive: If True, always ask user even if EVAL_PROVIDER is set

    Returns:
        Tuple of (provider, api_key) or None if no provider available
    """
    available = detect_available_providers()

    # No providers available
    if not available:
        console.print(get_missing_provider_message())
        return None

    # Check if EVAL_PROVIDER is already set and not forcing interactive
    if not force_interactive:
        env_provider = get_provider_from_env()
        if env_provider:
            config = PROVIDER_CONFIGS[env_provider]

            # Special case for Ollama - check if running, not env var
            if env_provider == LLMProvider.OLLAMA:
                if is_ollama_running():
                    console.print(f"[dim]Using {config.display_name} (from EVAL_PROVIDER)[/dim]")
                    return env_provider, "ollama"
                else:
                    console.print(f"[yellow]EVAL_PROVIDER=ollama but Ollama is not running[/yellow]")
                    console.print(f"[dim]Start Ollama with: ollama serve[/dim]")
                    # Fall through to interactive selection
            else:
                api_key = os.getenv(config.env_var)
                if api_key:
                    console.print(f"[dim]Using {config.display_name} (from EVAL_PROVIDER)[/dim]")
                    return env_provider, api_key
                else:
                    console.print(f"[yellow]EVAL_PROVIDER={env_provider.value} but {config.env_var} not set[/yellow]")
                    # Fall through to interactive selection

    # Only one provider available -> use it automatically
    if len(available) == 1 and not force_interactive:
        provider, api_key = available[0]
        config = PROVIDER_CONFIGS[provider]
        console.print(f"[dim]Using {config.display_name} (only available provider)[/dim]")
        return provider, api_key

    # Multiple providers available -> ask user to choose
    return interactive_provider_selection(console)
