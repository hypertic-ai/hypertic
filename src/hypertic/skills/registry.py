"""Skill registry for discovering skills from filesystem."""

from pathlib import Path

from hypertic.skills.base import SkillLoader, SkillMetadata
from hypertic.utils.log import get_logger

logger = get_logger(__name__)


class SkillRegistry:
    """Discovers and manages skills from filesystem."""

    DEFAULT_PATHS = [
        Path.home() / ".hypertic" / "skills",
        Path(".") / "skills",
    ]

    @staticmethod
    def discover_skills(search_paths: list[Path] | None = None) -> dict[str, SkillMetadata]:
        """
        Discover all skills from filesystem.

        Args:
            search_paths: Optional list of paths to search. Uses DEFAULT_PATHS if None.

        Returns:
            Dictionary mapping skill names to SkillMetadata
        """
        if search_paths is None:
            search_paths = SkillRegistry.DEFAULT_PATHS

        skills: dict[str, SkillMetadata] = {}

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for skill_dir in search_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                try:
                    skill = SkillLoader.load_skill(skill_dir, load_full=False)
                    skills[skill.name] = skill
                except Exception as e:
                    logger.warning(f"Failed to load skill from {skill_dir}: {e}")

        return skills
