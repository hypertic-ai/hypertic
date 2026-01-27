"""Base classes and loader for Skills."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hypertic.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class SkillMetadata:
    """Metadata and content for a Skill (loaded from filesystem)."""

    name: str
    description: str
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    path: Path | None = None
    instructions: str = ""
    tools: list[Any] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)

    def is_active(self) -> bool:
        return bool(self.instructions or self.tools)


class SkillLoader:
    """Loads Skills from filesystem."""

    @staticmethod
    def load_skill(path: str | Path, load_full: bool = False) -> SkillMetadata:
        """
        Load a skill from filesystem.

        Args:
            path: Path to skill directory (contains SKILL.md) or skill name
            load_full: If True, load full instructions and tools.
                      If False, load only metadata.

        Returns:
            SkillMetadata with loaded content
        """
        skill_path = SkillLoader._resolve_skill_path(path)
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill not found at: {skill_path}")

        skill_md_path = skill_path / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in skill directory: {skill_path}")

        with open(skill_md_path, encoding="utf-8") as f:
            content = f.read()

        frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"SKILL.md missing YAML frontmatter: {skill_md_path}")

        frontmatter = yaml.safe_load(frontmatter_match.group(1))
        instructions_body = content[frontmatter_match.end() :].strip()

        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")
        version = frontmatter.get("version", "1.0.0")
        tags = frontmatter.get("tags", [])

        if not name:
            raise ValueError(f"Skill missing 'name' in frontmatter: {skill_md_path}")
        if not description:
            raise ValueError(f"Skill missing 'description' in frontmatter: {skill_md_path}")

        metadata = SkillMetadata(
            name=name,
            description=description,
            version=version,
            tags=tags if isinstance(tags, list) else [],
            path=skill_path,
        )

        if load_full:
            metadata.instructions = instructions_body

            tools_path = skill_path / "tools.py"
            if tools_path.exists():
                try:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location("skill_tools", tools_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        tools = []
                        for attr_name in dir(module):
                            if attr_name.startswith("_"):
                                continue
                            attr = getattr(module, attr_name)
                            if hasattr(attr, "_tool_metadata") or (
                                hasattr(attr, "__class__")
                                and "BaseToolkit" in str(attr.__class__.__bases__)
                            ):
                                tools.append(attr)
                        metadata.tools = tools
                except Exception as e:
                    logger.warning(f"Failed to load tools from {tools_path}: {e}")

            for resource_file in ["REFERENCE.md", "EXAMPLES.md"]:
                resource_path = skill_path / resource_file
                if resource_path.exists():
                    with open(resource_path, encoding="utf-8") as f:
                        metadata.resources[resource_file] = f.read()

        return metadata

    @staticmethod
    def _resolve_skill_path(path: str | Path) -> Path:
        """Resolve skill path from string name or Path."""
        if isinstance(path, Path):
            return path.resolve()

        path_str = str(path)

        direct_path = Path(path_str).resolve()
        if direct_path.exists() and (direct_path / "SKILL.md").exists():
            return direct_path

        user_skills_path = Path.home() / ".hypertic" / "skills" / path_str
        if user_skills_path.exists() and (user_skills_path / "SKILL.md").exists():
            return user_skills_path.resolve()

        local_skills_path = Path(".").resolve() / "skills" / path_str
        if local_skills_path.exists() and (local_skills_path / "SKILL.md").exists():
            return local_skills_path.resolve()

        return Path(path_str).resolve()
