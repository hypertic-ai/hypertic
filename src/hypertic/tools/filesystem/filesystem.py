import fnmatch
import json
import os
import shutil
from pathlib import Path

from hypertic.tools.base import BaseToolkit, tool


class FileSystemTools(BaseToolkit):
    root_dir: str | None = None

    def _get_path(self, path: str) -> Path:
        if self.root_dir:
            full_path = Path(self.root_dir) / path
        else:
            full_path = Path(path).resolve()

        if self.root_dir:
            root = Path(self.root_dir).resolve()
            try:
                full_path.resolve().relative_to(root)
            except ValueError as err:
                raise ValueError(f"Path '{path}' is outside allowed directory: {err}") from err

        return full_path

    @tool
    def copy_file(self, source_path: str, destination_path: str) -> str:
        """Copy a file from source to destination.

        Creates a copy of a file in a specified location.

        Args:
            source_path: Path of the file to copy.
            destination_path: Path where the copied file should be saved.

        Returns:
            JSON string with success message or error details.
        """
        try:
            source = self._get_path(source_path)
            destination = self._get_path(destination_path)

            if not source.exists():
                raise ValueError(f"Source file does not exist: {source_path}")

            if source.is_dir():
                raise ValueError(f"Source path is a directory, not a file: {source_path}")

            destination.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source, destination, follow_symlinks=False)

            result = {
                "success": True,
                "message": f"File copied successfully from {source_path} to {destination_path}",
                "source": str(source),
                "destination": str(destination),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to copy file: {e}") from e

    @tool
    def delete_file(self, file_path: str) -> str:
        """Delete a file.

        Removes a file from the file system.

        Args:
            file_path: Path of the file to delete.

        Returns:
            JSON string with success message or error details.
        """
        try:
            path = self._get_path(file_path)

            if not path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            if path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {file_path}")

            path.unlink()

            result = {
                "success": True,
                "message": f"File deleted successfully: {file_path}",
                "file_path": str(path),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to delete file: {e}") from e

    @tool
    def search_files(self, pattern: str, dir_path: str = ".") -> str:
        """Search for files matching a pattern.

        Recursively searches for files in a directory that match a Unix shell pattern.

        Args:
            pattern: Unix shell pattern (e.g., "*.py", "test_*.txt").
            dir_path: Directory to search in (default: current directory).

        Returns:
            JSON string containing list of matching file paths.
        """
        try:
            search_dir = self._get_path(dir_path)

            if not search_dir.exists():
                raise ValueError(f"Directory does not exist: {dir_path}")

            if not search_dir.is_dir():
                raise ValueError(f"Path is not a directory: {dir_path}")

            matches = []
            root_resolved = Path(self.root_dir).resolve() if self.root_dir else search_dir.resolve()
            for root, _, filenames in os.walk(search_dir):
                for filename in fnmatch.filter(filenames, pattern):
                    absolute_path = Path(root) / filename
                    absolute_path_resolved = absolute_path.resolve()
                    root_resolved_path = root_resolved.resolve()

                    try:
                        if self.root_dir:
                            relative_path = absolute_path_resolved.relative_to(root_resolved_path)
                        else:
                            relative_path = absolute_path_resolved.relative_to(search_dir.resolve())
                        matches.append(str(relative_path))
                    except ValueError:
                        continue

            result = {
                "pattern": pattern,
                "search_directory": str(search_dir),
                "count": len(matches),
                "matches": matches,
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to search files: {e!s}") from e

    @tool
    def move_file(self, source_path: str, destination_path: str) -> str:
        """Move or rename a file.

        Moves a file from one location to another. Can also be used to rename files.

        Args:
            source_path: Path of the file to move.
            destination_path: New path for the moved file.

        Returns:
            JSON string with success message or error details.
        """
        try:
            source = self._get_path(source_path)
            destination = self._get_path(destination_path)

            if not source.exists():
                raise ValueError(f"Source file does not exist: {source_path}")

            if source.is_dir():
                raise ValueError(f"Source path is a directory, not a file: {source_path}")

            destination.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source), destination)

            result = {
                "success": True,
                "message": f"File moved successfully from {source_path} to {destination_path}",
                "source": str(source),
                "destination": str(destination),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to move file: {e}") from e

    @tool
    def read_file(self, file_path: str) -> str:
        """Read the contents of a file.

        Reads and returns the entire contents of a text file.

        Args:
            file_path: Path of the file to read.

        Returns:
            JSON string containing file contents and metadata.
        """
        try:
            path = self._get_path(file_path)

            if not path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            if path.is_dir():
                raise ValueError(f"Path is a directory, not a file: {file_path}")

            with open(path, encoding="utf-8") as f:
                content = f.read()

            result = {"file_path": str(path), "size": len(content), "content": content}

            return json.dumps(result, indent=2)
        except UnicodeDecodeError as err:
            raise ValueError(f"File '{file_path}' is not a text file (binary content)") from err
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}") from e

    @tool
    def write_file(self, file_path: str, text: str, append: bool = False) -> str:
        """Write text to a file.

        Writes text content to a file. Can append to existing file or overwrite.

        Args:
            file_path: Path of the file to write to.
            text: Text content to write.
            append: Whether to append to existing file (default: False, overwrites).

        Returns:
            JSON string with success message or error details.
        """
        try:
            path = self._get_path(file_path)

            path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(text)

            action = "appended to" if append else "written to"
            result = {
                "success": True,
                "message": f"File {action} successfully: {file_path}",
                "file_path": str(path),
                "size": len(text),
                "append": append,
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to write file: {e!s}") from e

    @tool
    def list_directory(self, dir_path: str = ".") -> str:
        """List files and directories in a directory.

        Lists all files and directories in the specified directory.
        When root_dir is set, "." refers to the root directory.

        Args:
            dir_path: Directory to list (default: "." for current/root directory).

        Returns:
            JSON string containing list of files and directories.
        """
        try:
            path = self._get_path(dir_path)

            if not path.exists():
                raise ValueError(f"Directory does not exist: {dir_path}")

            if not path.is_dir():
                raise ValueError(f"Path is not a directory: {dir_path}")

            entries = []
            for entry in sorted(path.iterdir()):
                entry_info = {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "path": str(entry.relative_to(path)),
                }
                entries.append(entry_info)

            result = {"directory": str(path), "count": len(entries), "entries": entries}

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}") from e
