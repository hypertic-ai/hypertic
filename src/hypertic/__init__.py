from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hypertic")
except PackageNotFoundError:
    __version__ = "0.0.1"

__all__ = ["__version__"]
