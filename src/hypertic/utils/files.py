import base64
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, model_validator

from hypertic.utils.log import get_logger

logger = get_logger(__name__)


class FileType(Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class File(BaseModel):
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[bytes] = None

    file_type: Optional[FileType] = None
    mime_type: Optional[str] = None
    format: Optional[str] = None

    id: Optional[str] = None
    filename: Optional[str] = None
    size: Optional[int] = None
    description: Optional[str] = None
    file_id: Optional[str] = None

    detail: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            url = data.get("url")
            filepath = data.get("filepath")
            content = data.get("content")
            source = data.get("source")

            if source and not url and not filepath and not content:
                if source.startswith(("http://", "https://")):
                    data["url"] = source
                else:
                    data["filepath"] = source
                if "source" in data:
                    del data["source"]

            url = data.get("url")
            filepath = data.get("filepath")
            content = data.get("content")

            sources = [x for x in [url, filepath, content] if x is not None]
            if len(sources) == 0:
                raise ValueError("One of 'url', 'filepath', or 'content' must be provided")
            elif len(sources) > 1:
                raise ValueError("Only one of 'url', 'filepath', or 'content' should be provided")

            if data.get("id") is None:
                data["id"] = str(uuid4())

            if data.get("file_type") is None:
                source_str = url or filepath or ""
                if source_str:
                    data["file_type"] = cls._detect_type(source_str)

            if data.get("mime_type") is None:
                source_str = url or filepath or ""
                if source_str:
                    guessed_type, _ = mimetypes.guess_type(str(source_str))
                    if guessed_type:
                        data["mime_type"] = guessed_type

            if data.get("filename") is None:
                source_str = url or filepath or ""
                if source_str:
                    if isinstance(source_str, Path):
                        data["filename"] = source_str.name
                    elif "/" in str(source_str):
                        data["filename"] = str(source_str).split("/")[-1]
                    elif "\\" in str(source_str):
                        data["filename"] = str(source_str).split("\\")[-1]

        return data

    @staticmethod
    def _detect_type(source: str) -> FileType:
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"]

        audio_extensions = [".wav", ".mp3", ".m4a", ".mp4", ".mpga", ".mpeg", ".mpg", ".webm"]

        video_extensions = [
            ".mp4",
            ".mpeg",
            ".mov",
            ".avi",
            ".flv",
            ".mpg",
            ".webm",
            ".wmv",
            ".3gpp",
        ]

        document_extensions = [
            ".pdf",
            ".txt",
            ".md",
            ".doc",
            ".docx",
            ".rtf",
            ".odt",
            ".epub",
            ".ppt",
            ".pptx",
            ".csv",
            ".tsv",
            ".xls",
            ".xlsx",
            ".json",
            ".xml",
            ".yaml",
            ".html",
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
        ]

        source_lower = str(source).lower()

        if source.startswith(("http://", "https://")):
            if "youtube.com/watch" in source_lower or "youtu.be/" in source_lower:
                return FileType.VIDEO

            if any(ext in source_lower for ext in image_extensions):
                return FileType.IMAGE
            elif any(ext in source_lower for ext in audio_extensions):
                return FileType.AUDIO
            elif any(ext in source_lower for ext in video_extensions):
                return FileType.VIDEO
            elif any(ext in source_lower for ext in document_extensions):
                return FileType.DOCUMENT
            else:
                return FileType.FILE
        else:
            if any(source_lower.endswith(ext) for ext in image_extensions):
                return FileType.IMAGE
            elif any(source_lower.endswith(ext) for ext in audio_extensions):
                return FileType.AUDIO
            elif any(source_lower.endswith(ext) for ext in video_extensions):
                return FileType.VIDEO
            elif any(source_lower.endswith(ext) for ext in document_extensions):
                return FileType.DOCUMENT
            else:
                return FileType.FILE

    def get_content_bytes(self) -> Optional[bytes]:
        if self.content:
            return self.content
        elif self.url:
            try:
                import requests

                response = requests.get(self.url)
                response.raise_for_status()
                return bytes(response.content)
            except Exception:
                return None
        elif self.filepath:
            try:
                with open(self.filepath, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None

    def to_base64(self) -> Optional[str]:
        content_bytes = self.get_content_bytes()
        if content_bytes:
            return base64.b64encode(content_bytes).decode("utf-8")
        return None

    @classmethod
    def from_base64(
        cls,
        base64_content: str,
        id: Optional[str] = None,
        mime_type: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> "File":
        try:
            content_bytes = base64.b64decode(base64_content)
        except Exception:
            content_bytes = base64_content.encode("utf-8")

        return cls(
            content=content_bytes,
            id=id or str(uuid4()),
            mime_type=mime_type,
            format=format,
            **kwargs,
        )

    def to_dict(self, include_base64_content: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "file_type": self.file_type.value if self.file_type else None,
            "mime_type": self.mime_type,
            "format": self.format,
            "filename": self.filename,
            "size": self.size,
            "description": self.description,
            "file_id": self.file_id,
            "detail": self.detail,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

        if include_base64_content:
            base64_data = self.to_base64()
            if base64_data:
                result["content"] = base64_data

        return {k: v for k, v in result.items() if v is not None}


class FileProcessor:
    @staticmethod
    def process_files(file_sources: list[str]) -> list[File]:
        files = []
        for source in file_sources:
            try:
                file_obj = File(
                    url=source if source.startswith(("http://", "https://")) else None,
                    filepath=source if not source.startswith(("http://", "https://")) else None,
                )
                files.append(file_obj)
            except Exception as e:
                logger.warning(f"Could not process file {source}: {e}", exc_info=True)
        return files

    @staticmethod
    def process_message(message: dict[str, Any], provider: str) -> dict[str, Any]:
        if "files" not in message:
            return message

        file_objects = FileProcessor.process_files(message["files"])

        message["files"] = [f.to_dict(include_base64_content=False) for f in file_objects]
        message["_file_objects"] = file_objects

        return message

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        guessed_type, _ = mimetypes.guess_type(file_path)
        if guessed_type:
            return guessed_type

        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".pdf"):
            return "application/pdf"
        elif file_path_lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif file_path_lower.endswith(".png"):
            return "image/png"
        elif file_path_lower.endswith(".gif"):
            return "image/gif"
        elif file_path_lower.endswith(".webp"):
            return "image/webp"
        elif file_path_lower.endswith(".mp3"):
            return "audio/mpeg"
        elif file_path_lower.endswith(".wav"):
            return "audio/wav"
        elif file_path_lower.endswith(".mp4"):
            return "video/mp4"
        elif file_path_lower.endswith(".txt"):
            return "text/plain"
        else:
            return "application/octet-stream"
