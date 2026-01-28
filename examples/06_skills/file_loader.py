import json
from pathlib import Path

from hypertic.tools.base import BaseToolkit, tool


class FileLoaderTools(BaseToolkit):
    """Tools for loading and reading JSON files."""

    path: str

    def _get_full_path(self, file_path: str | None = None) -> Path:
        """Get the full path, using path as default if no file_path provided."""
        if file_path is None:
            file_path = self.path

        path = Path(file_path)
        if not path.is_absolute():
            path = path.resolve()
        return path

    @tool
    def load_json_file(self, file_path: str | None = None) -> str:
        """Load and parse a JSON file, returning its content as formatted text.

        This is especially useful for JSON files that can't be attached as documents
        (e.g., with Anthropic API which only supports PDFs). The tool reads the file,
        parses it as JSON, and returns it as formatted text that can be used in analysis.

        If no file_path is provided, uses the input file path configured in the tool.

        Args:
            file_path: Optional path to the JSON file to load. If not provided, uses the input file.

        Returns:
            Formatted JSON string with the file contents

        Example:
            load_json_file()  # Uses the input file path
            load_json_file("data/other_file.json")  # Loads a different file
        """
        try:
            path = self._get_full_path(file_path)

            if not path.exists():
                return json.dumps(
                    {"error": f"File not found: {file_path or self.path}", "resolved_path": str(path), "input_file": self.path}, indent=2
                )

            if not path.suffix.lower() == ".json":
                return json.dumps({"error": f"File is not a JSON file: {file_path or self.path}", "file_extension": path.suffix}, indent=2)

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            return json.dumps(data, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": f"Invalid JSON in file: {file_path or self.path}",
                    "details": str(e),
                    "resolved_path": str(path) if "path" in locals() else None,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error loading file: {file_path or self.path}",
                    "details": str(e),
                    "resolved_path": str(path) if "path" in locals() else None,
                },
                indent=2,
            )
