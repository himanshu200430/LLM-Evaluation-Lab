"""
Which models can be used: Gemini, Groq, and any model you add yourself.

Custom models are saved in backend/custom_models.json.
No API keys are stored there, only the NAME of the .env variable
that holds the key.
"""

import json
from pathlib import Path

from app.providers.custom import CustomProvider


CUSTOM_FILE = Path(__file__).resolve().parents[2] / "custom_models.json"

BUILT_IN = ["Gemini", "Groq"]


# Preset name -> (base URL, usual name of the .env key).
# Check the provider's docs if one fails.
PRESETS = {
    "Groq": (
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
    ),
    "Gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
    ),
    "OpenRouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
    ),
    "OpenAI": (
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
    ),
    "DeepSeek": (
        "https://api.deepseek.com",
        "DEEPSEEK_API_KEY",
    ),
    "Together": (
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
    ),
    "Ollama (local)": (
        "http://localhost:11434/v1",
        "",
    ),
}


def load_custom_specs():
    if not CUSTOM_FILE.exists():
        return []

    try:
        return json.loads(
            CUSTOM_FILE.read_text(encoding="utf-8")
        )

    except json.JSONDecodeError:
        return []


def validate_spec(spec):
    """Check and clean what the user typed. Raises ValueError with a clear message."""

    name = str(spec.get("name", "")).strip()
    model = str(spec.get("model", "")).strip()
    base_url = str(spec.get("base_url", "")).strip()

    if not name:
        raise ValueError("Give the model a display name.")

    if name in BUILT_IN:
        raise ValueError(
            f"'{name}' is a built-in name. Choose another display name."
        )

    if not model:
        raise ValueError("Enter the model id.")

    if not base_url.startswith(("http://", "https://")):
        raise ValueError(
            "The base URL must start with http:// or https://"
        )

    try:
        price_in = float(spec.get("price_in", 0))
        price_out = float(spec.get("price_out", 0))

    except (TypeError, ValueError):
        raise ValueError("Prices must be numbers.")

    if price_in < 0 or price_out < 0:
        raise ValueError("Prices cannot be negative.")

    return {
        "name": name,
        "model": model,
        "base_url": base_url,
        "api_key_env": str(
            spec.get("api_key_env", "")
        ).strip(),
        "price_in": price_in,
        "price_out": price_out,
    }


def save_custom_spec(spec):
    """Add a model or replace the one with the same name."""

    clean = validate_spec(spec)

    specs = [
        spec
        for spec in load_custom_specs()
        if spec["name"] != clean["name"]
    ] + [clean]

    CUSTOM_FILE.write_text(
        json.dumps(specs, indent=2),
        encoding="utf-8",
    )

    return clean


def delete_custom_spec(name):
    specs = [
        spec
        for spec in load_custom_specs()
        if spec["name"] != name
    ]

    CUSTOM_FILE.write_text(
        json.dumps(specs, indent=2),
        encoding="utf-8",
    )


def all_model_names():
    return BUILT_IN + [
        spec["name"]
        for spec in load_custom_specs()
    ]


def build_provider(name):
    if name == "Gemini":
        from app.providers.gemini import GeminiProvider

        return GeminiProvider()

    if name == "Groq":
        from app.providers.groq import GroqProvider

        return GroqProvider()

    for spec in load_custom_specs():
        if spec["name"] == name:
            return CustomProvider(**spec)

    raise ValueError(f"Unknown model: {name}")


def build_providers(names):
    return [
        build_provider(name)
        for name in names
    ]