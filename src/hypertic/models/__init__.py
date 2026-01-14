from importlib import import_module
from typing import TYPE_CHECKING

from hypertic.models.base import Base, LLMResponse
from hypertic.models.metrics import Metrics

if TYPE_CHECKING:
    from hypertic.models.anthropic.anthropic import Anthropic
    from hypertic.models.cohere.cohere import Cohere
    from hypertic.models.deepseek.deepseek import DeepSeek
    from hypertic.models.fireworks.fireworks import FireworksAI
    from hypertic.models.google.google import GoogleAI
    from hypertic.models.groq.groq import Groq
    from hypertic.models.mistral.mistral import MistralAI
    from hypertic.models.moonshot.moonshot import MoonshotAI
    from hypertic.models.openai.openaichat import OpenAIChat
    from hypertic.models.openai.openairesponse import OpenAIResponse
    from hypertic.models.openrouter.openrouter import OpenRouter
    from hypertic.models.qwen.qwen import Qwen
    from hypertic.models.xai.xai import XAI


_module_lookup = {
    "Anthropic": "hypertic.models.anthropic.anthropic",
    "Cohere": "hypertic.models.cohere.cohere",
    "DeepSeek": "hypertic.models.deepseek.deepseek",
    "FireworksAI": "hypertic.models.fireworks.fireworks",
    "GoogleAI": "hypertic.models.google.google",
    "Groq": "hypertic.models.groq.groq",
    "MistralAI": "hypertic.models.mistral.mistral",
    "MoonshotAI": "hypertic.models.moonshot.moonshot",
    "OpenAIChat": "hypertic.models.openai.openaichat",
    "OpenAIResponse": "hypertic.models.openai.openairesponse",
    "OpenRouter": "hypertic.models.openrouter.openrouter",
    "Qwen": "hypertic.models.qwen.qwen",
    "XAI": "hypertic.models.xai.xai",
}


def __getattr__(name: str):
    if name in _module_lookup:
        module_path = _module_lookup[name]
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_lookup.keys()) + ["Base", "LLMResponse", "Metrics"]
