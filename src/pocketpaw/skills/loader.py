"""
SkillLoader - Load and parse skills from the AgentSkills ecosystem.

Skills are loaded from:
1. ~/.agents/skills/ - Central location (installed via `npx skills add`)
2. ~/.pocketpaw/skills/ - PocketPaw-specific skills

Skills follow the AgentSkills spec: a directory with SKILL.md containing
YAML frontmatter and markdown instructions.

Progressive disclosure support:
  Each skill exposes a build_summary() for Pass 1 (description + sub-file
  manifest only) and load_sub_file() for on-demand content loading in Pass 2.

Multi-file skill support:
  SKILL.md frontmatter may declare a `files:` key mapping filenames to
  one-line descriptions. These sub-files live in the same directory and are
  loaded on demand by SkillExecutor — never upfront.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Skill search paths in priority order (later overrides earlier)
SKILL_PATHS = [
    Path.home() / ".agents" / "skills",  # From skills.sh (central)
    Path.home() / ".claude" / "skills",  # Claude Code / SDK standard
    Path.home() / ".pocketpaw" / "skills",  # PocketPaw-specific
]


@dataclass
class Skill:
    """
    Represents a loaded skill.

    The `content` field holds the body of SKILL.md — the always-loaded core
    instructions.  Sub-files are declared in `sub_files` but their content is
    NOT stored here; call load_sub_file() to fetch them on demand.
    """

    name: str
    description: str
    content: str  # Body of SKILL.md (always loaded)
    path: Path  # Path to SKILL.md

    # Optional frontmatter fields
    user_invocable: bool = True
    disable_model_invocation: bool = False
    argument_hint: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Progressive disclosure / multi-file support
    # Maps filename → one-line description (declared in frontmatter `files:`)
    sub_files: dict[str, str] = field(default_factory=dict)

    def build_prompt(self, args: str = "") -> str:
        """
        Build the full prompt from SKILL.md body.

        Substitutes argument placeholders:
          $ARGUMENTS  → full args string
          $0, $1, …   → positional args split on whitespace
        """
        prompt = self.content
        arg_list = args.split() if args else []

        prompt = prompt.replace("$ARGUMENTS", args)
        for i, arg in enumerate(arg_list):
            prompt = prompt.replace(f"${i}", arg)

        return prompt

    def build_summary(self) -> str:
        """
        Return a compact summary suitable for progressive disclosure Pass 1.

        Contains only: name, description, argument_hint, and the sub-file
        manifest (filenames + one-line descriptions).  No skill body content
        is included, keeping the token cost minimal.
        """
        lines: list[str] = [
            f"Skill: {self.name}",
            f"Description: {self.description}",
        ]

        if self.argument_hint:
            lines.append(f"Usage: {self.argument_hint}")

        if self.sub_files:
            lines.append("Sub-files:")
            for fname, desc in self.sub_files.items():
                lines.append(f"  - {fname}: {desc}")
        else:
            lines.append("(No sub-files — request __full__ to load complete skill body)")

        return "\n".join(lines)

    def load_sub_file(self, filename: str) -> str | None:
        """
        Load the raw content of a declared sub-file.

        Args:
            filename: Bare filename (e.g. "techniques.md"), must be present
                      in self.sub_files.

        Returns:
            File content as a string, or None if the file cannot be read.
        """
        if filename not in self.sub_files:
            logger.warning(
                f"[{self.name}] Requested sub-file '{filename}' is not declared "
                f"in frontmatter. Declared: {list(self.sub_files)}"
            )
            return None

        target = self.path.parent / filename
        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to read sub-file '{filename}': {e}")
            return None

    def load_sub_files(self, filenames: list[str]) -> dict[str, str]:
        """
        Load multiple sub-files at once.

        Args:
            filenames: List of filenames to load.

        Returns:
            Dict mapping filename → content for successfully loaded files.
            Files that fail to load are omitted (errors are logged).
        """
        result: dict[str, str] = {}
        for fname in filenames:
            content = self.load_sub_file(fname)
            if content is not None:
                result[fname] = content
        return result


def parse_skill_md(skill_path: Path) -> "Skill | None":
    """
    Parse a SKILL.md file into a Skill object.

    Frontmatter fields recognised:
      name                  str      — skill identifier (falls back to dir name)
      description           str      — one-line summary
      user-invocable        bool     — expose as slash command (default: true)
      disable-model-invocation bool  — (default: false)
      argument-hint         str      — usage example shown in /help
      allowed-tools         list     — tool whitelist
      metadata              dict     — arbitrary extra data
      files                 dict     — sub-file manifest: {filename: description}

    The `files:` key is the new addition.  Example:

        files:
          injection-patterns.md: Common injection vectors and detection
          auth-checklist.md: Authentication and authorization review steps

    Sub-files that are declared but do not exist on disk are silently dropped
    with a warning so a typo in frontmatter doesn't break the whole skill.

    Args:
        skill_path: Path to SKILL.md file

    Returns:
        Skill object, or None if parsing fails.
    """
    try:
        text = skill_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {skill_path}: {e}")
        return None

    # Extract YAML frontmatter between --- markers
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        logger.warning(f"No frontmatter found in {skill_path}")
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        content = match.group(2).strip()
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML in {skill_path}: {e}")
        return None

    # -- Name
    name: str = frontmatter.get("name") or skill_path.parent.name

    # -- Description
    description: str = frontmatter.get("description", "")

    # -- Sub-files
    # `files:` in frontmatter is a dict mapping filename → one-line description.
    # We validate each entry exists on disk; missing files are dropped.
    declared_files: dict = frontmatter.get("files") or {}
    skill_dir = skill_path.parent
    sub_files: dict[str, str] = {}

    for fname, fdesc in declared_files.items():
        fname = str(fname)
        fdesc = str(fdesc) if fdesc else ""
        candidate = skill_dir / fname

        if not candidate.exists():
            logger.warning(
                f"[{name}] Declared sub-file '{fname}' does not exist at {candidate} — skipping."
            )
            continue

        if not candidate.is_file():
            logger.warning(
                f"[{name}] Declared sub-file '{fname}' is not a regular file — skipping."
            )
            continue

        sub_files[fname] = fdesc
        logger.debug(f"[{name}] Registered sub-file: {fname}")

    if sub_files:
        logger.debug(f"[{name}] {len(sub_files)} sub-file(s) registered")

    return Skill(
        name=name,
        description=description,
        content=content,
        path=skill_path,
        user_invocable=frontmatter.get("user-invocable", True),
        disable_model_invocation=frontmatter.get("disable-model-invocation", False),
        argument_hint=frontmatter.get("argument-hint"),
        allowed_tools=frontmatter.get("allowed-tools", []),
        metadata=frontmatter.get("metadata", {}),
        sub_files=sub_files,
    )


class SkillLoader:
    """
    Loads skills from configured paths.

    Supports hot-reloading when skills change on disk.

    Skills are discovered by scanning each path in SKILL_PATHS for
    subdirectories that contain a SKILL.md file.  Later paths in
    SKILL_PATHS override earlier ones (priority order).
    """

    def __init__(self, extra_paths: list[Path] | None = None):
        """
        Initialise the skill loader.

        Args:
            extra_paths: Additional paths to search for skills (appended after
                         the default SKILL_PATHS, so they take highest priority).
        """
        self.paths = SKILL_PATHS.copy()
        if extra_paths:
            self.paths.extend(extra_paths)

        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def load(self, force: bool = False) -> dict[str, Skill]:
        """
        Load all skills from configured paths.

        Args:
            force: Force reload even if already loaded.

        Returns:
            Dict mapping skill names to Skill objects.
        """
        if self._loaded and not force:
            return self._skills

        self._skills = {}

        for base_path in self.paths:
            if not base_path.exists():
                continue

            logger.debug(f"Scanning for skills in {base_path}")

            for item in base_path.iterdir():
                # Handle both directories and symlinks to directories
                if not item.is_dir():
                    continue

                skill_md = item / "SKILL.md"
                if not skill_md.exists():
                    continue

                skill = parse_skill_md(skill_md)
                if skill:
                    # Later paths override earlier (priority order)
                    self._skills[skill.name] = skill
                    logger.debug(f"Loaded skill: {skill.name}")

        self._loaded = True
        logger.info(f"Loaded {len(self._skills)} skills")
        return self._skills

    def reload(self) -> dict[str, Skill]:
        """Force reload all skills from disk."""
        return self.load(force=True)

    def get(self, name: str) -> "Skill | None":
        """
        Get a skill by name.

        Args:
            name: Skill name (e.g. "find-skills").

        Returns:
            Skill object or None if not found.
        """
        if not self._loaded:
            self.load()
        return self._skills.get(name)

    def get_all(self) -> dict[str, "Skill"]:
        """Get all loaded skills."""
        if not self._loaded:
            self.load()
        return self._skills.copy()

    def get_invocable(self) -> list["Skill"]:
        """Get all user-invocable skills (for slash commands)."""
        if not self._loaded:
            self.load()
        return [s for s in self._skills.values() if s.user_invocable]

    def search(self, query: str = "") -> list["Skill"]:
        """
        Search user-invocable skills by name and description.

        Args:
            query: Case-insensitive substring to match against name +
                   description.  Empty string returns all invocable skills.

        Returns:
            List of matching Skill objects.
        """
        invocable = self.get_invocable()
        if not query:
            return invocable
        q = query.lower()
        return [s for s in invocable if q in s.name.lower() or q in s.description.lower()]

    def list_names(self) -> list[str]:
        """Get list of all loaded skill names."""
        if not self._loaded:
            self.load()
        return list(self._skills.keys())


_skill_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    """Get the singleton SkillLoader instance."""
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
    return _skill_loader


def load_all_skills() -> dict[str, "Skill"]:
    """Convenience function to load all skills."""
    return get_skill_loader().load()
