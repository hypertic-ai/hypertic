from importlib import import_module
from typing import TYPE_CHECKING

from hypertic.tools.base import BaseToolkit, tool

if TYPE_CHECKING:
    from hypertic.tools.dalle.dalle import DalleTools
    from hypertic.tools.duckduckgo.duckduckgo import DuckDuckGoTools
    from hypertic.tools.filesystem.filesystem import FileSystemTools
    from hypertic.tools.mcp.client import ExecutableTool, MCPServers

_module_lookup = {
    "DalleTools": "hypertic.tools.dalle.dalle",
    "DuckDuckGoTools": "hypertic.tools.duckduckgo.duckduckgo",
    "FileSystemTools": "hypertic.tools.filesystem.filesystem",
    "MCPServers": "hypertic.tools.mcp.client",
    "ExecutableTool": "hypertic.tools.mcp.client",
}


def __getattr__(name: str):
    if name in _module_lookup:
        module_path = _module_lookup[name]
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_lookup.keys()) + ["BaseToolkit", "tool"]
