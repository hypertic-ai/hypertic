from importlib import import_module
from typing import TYPE_CHECKING

from hypertic.embedders.base import BaseEmbedder

if TYPE_CHECKING:
    from hypertic.embedders.cohere.cohere import CohereEmbedder
    from hypertic.embedders.google.google import GoogleEmbedder
    from hypertic.embedders.huggingface.huggingface import HuggingFaceEmbedder
    from hypertic.embedders.mistral.mistral import MistralEmbedder
    from hypertic.embedders.openai.openai import OpenAIEmbedder
    from hypertic.embedders.sentencetransformer.sentencetransformer import SentenceTransformerEmbedder


_module_lookup = {
    "CohereEmbedder": "hypertic.embedders.cohere.cohere",
    "GoogleEmbedder": "hypertic.embedders.google.google",
    "HuggingFaceEmbedder": "hypertic.embedders.huggingface.huggingface",
    "MistralEmbedder": "hypertic.embedders.mistral.mistral",
    "OpenAIEmbedder": "hypertic.embedders.openai.openai",
    "SentenceTransformerEmbedder": "hypertic.embedders.sentencetransformer.sentencetransformer",
}


def __getattr__(name: str):
    if name in _module_lookup:
        module_path = _module_lookup[name]
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_lookup.keys()) + ["BaseEmbedder"]
