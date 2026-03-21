"""
SkillExecutor - Execute skills through the AgentRouter.

Handles:
1. Building the prompt from skill content + user args
2. Running through the configured agent (Open Interpreter / Claude Code)
3. Streaming results back

Progressive disclosure (two-pass loading)
-----------------------------------------
When `direct=False` (the default for agent-selected skills), the executor
runs a two-pass flow to save context window:

  Pass 1 — inject skill name, description, argument_hint, and sub-file
            manifest (if any) for all candidate skills.  The agent responds
            with a JSON list of filenames it needs, or ["__full__"] to load
            the complete SKILL.md body, or [] if the summary alone is enough.

  Pass 2 — load only the selected files and execute the real task.

When `direct=True` (user explicitly typed e.g. /commit), Pass 1 is skipped
and the full skill content is injected immediately.  This avoids a wasted
round-trip when the caller already knows exactly what to run.

Multi-file skill support
------------------------
If a skill declares sub-files in its SKILL.md frontmatter, Pass 1 shows the
manifest.  Pass 2 loads only the agent-selected sub-files via
Skill.load_sub_file().  The SKILL.md body (core instructions) is always
appended when sub-files are loaded, giving the agent the execution context it
needs.

AgentRouter contract
--------------------
SkillExecutor calls two methods on AgentRouter:

  router.run(prompt: str) -> AsyncIterator[dict]
    Streams execution chunks (unchanged from original).

  router.run_json(prompt: str) -> list[str]
    Returns a parsed JSON list.  Must be awaited.  If your AgentRouter does
    not yet expose this method, add a thin wrapper that calls run() and
    extracts the first JSON array from the accumulated text output.
    A reference implementation is provided in _AgentRouterJsonMixin below.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..agents.router import AgentRouter
from ..config import Settings, get_settings
from .loader import Skill, SkillLoader, get_skill_loader

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Sentinel value the agent returns when it wants the full SKILL.md body
_FULL_SENTINEL = "__full__"

# System prompt fragment shown to the agent during Pass 1
_DISCLOSURE_SYSTEM = (
    "You are a skill-selection assistant. "
    "Given a skill summary and a user request, decide which sub-files (if any) "
    "are needed to complete the task. "
    'Reply with ONLY a JSON array of filenames, e.g. ["auth-checklist.md"]. '
    f'Use ["{_FULL_SENTINEL}"] to request the complete skill body. '
    "Use [] if the summary alone is sufficient. "
    "Do not include any explanation or markdown — raw JSON only."
)


# Mixin: run_json helper (attach to AgentRouter if it doesn't have it yet)


class _AgentRouterJsonMixin:
    """
    Reference mixin that adds run_json() on top of an existing run() method.

    If AgentRouter already exposes run_json(), this mixin is not needed.
    Paste the method body into AgentRouter directly, or subclass it.
    """

    async def run_json(self, prompt: str) -> list[str]:
        """
        Run a prompt and extract the first JSON array from the response.

        Returns an empty list on parse failure (safe degradation: executor
        will fall back to loading the full skill body).
        """
        chunks: list[str] = []
        async for chunk in self.run(prompt):  # type: ignore[attr-defined]
            if chunk.get("type") == "text":
                chunks.append(chunk.get("content", ""))

        raw = "".join(chunks).strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(item) for item in result]
            logger.warning(f"run_json: expected list, got {type(result).__name__}")
            return [_FULL_SENTINEL]
        except json.JSONDecodeError as e:
            logger.warning(f"run_json: JSON parse failed ({e}), defaulting to __full__")
            return [_FULL_SENTINEL]


# SkillExecutor                                                                #


class SkillExecutor:
    """
    Executes skills through the agent backend.

    Two execution modes
    -------------------
    direct=True   Skip progressive disclosure; inject full skill immediately.
                  Use for explicit slash-command invocations (/commit, /review).

    direct=False  Two-pass flow: show summary → agent selects → load selected.
                  Use when the agent is choosing from multiple skills, or when
                  context-window efficiency matters.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        skill_loader: SkillLoader | None = None,
    ):
        """
        Initialise the executor.

        Args:
            settings:     Settings instance (uses singleton if not provided).
            skill_loader: SkillLoader instance (uses singleton if not provided).
        """
        self.settings = settings or get_settings()
        self.skill_loader = skill_loader or get_skill_loader()

        # Agent router (created lazily)
        self._agent_router: AgentRouter | None = None

    # Internal: agent router

    def _get_agent_router(self) -> AgentRouter:
        """Get or create the agent router."""
        if self._agent_router is None:
            self._agent_router = AgentRouter(self.settings)
        return self._agent_router

    async def execute(
        self,
        skill_name: str,
        args: str = "",
        direct: bool = True,
    ) -> AsyncIterator[dict]:
        """
        Execute a skill by name.

        Args:
            skill_name: Name of the skill to execute.
            args:       Arguments to pass to the skill.
            direct:     If True, skip progressive disclosure (fast path).
                        Defaults to True for named invocations.

        Yields:
            Chunks from the agent execution.
        """
        skill = self.skill_loader.get(skill_name)
        if not skill:
            yield {
                "type": "error",
                "content": f"Skill not found: {skill_name}",
            }
            return

        async for chunk in self.execute_skill(skill, args=args, direct=direct):
            yield chunk

    async def execute_skill(
        self,
        skill: Skill,
        args: str = "",
        direct: bool = False,
    ) -> AsyncIterator[dict]:
        """
        Execute a Skill object.

        Args:
            skill:  Skill object to execute.
            args:   Arguments to pass to the skill.
            direct: If True, skip progressive disclosure and load the full
                    skill body immediately (best for explicit /slash commands).
                    If False, run the two-pass disclosure flow (saves tokens
                    when routing between multiple candidate skills).

        Yields:
            Chunks from the agent execution.
        """
        logger.info(f"Executing skill: {skill.name!r} | args={args!r} | direct={direct}")

        yield {
            "type": "skill_started",
            "skill_name": skill.name,
            "args": args,
            "direct": direct,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        try:
            if direct:
                # ── Fast path ────────────────────────────────────────────
                # User explicitly invoked this skill; load full content now.
                async for chunk in self._execute_full(skill, args):
                    yield chunk
            else:
                # ── Two-pass progressive disclosure ──────────────────────
                async for chunk in self._execute_with_disclosure(skill, args):
                    yield chunk

            yield {
                "type": "skill_completed",
                "skill_name": skill.name,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error executing skill {skill.name!r}: {e}", exc_info=True)
            yield {
                "type": "skill_error",
                "skill_name": skill.name,
                "error": str(e),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

    async def _execute_full(
        self,
        skill: Skill,
        args: str,
    ) -> AsyncIterator[dict]:
        """
        Build the full prompt (SKILL.md body) and execute it immediately.

        Used for direct invocations and as the fallback when the agent
        requests __full__ during progressive disclosure.
        """
        prompt = skill.build_prompt(args)
        full_prompt = self._wrap_prompt(skill.name, skill.description, prompt, args)

        logger.debug(f"[{skill.name}] Full prompt ({len(full_prompt)} chars)")

        async for chunk in self._get_agent_router().run(full_prompt):
            yield chunk

    async def _execute_with_disclosure(
        self,
        skill: Skill,
        args: str,
    ) -> AsyncIterator[dict]:
        """
        Two-pass execution flow.

        Pass 1: Show the agent only the skill summary + sub-file manifest.
                Ask it which files it needs.
        Pass 2: Load the selected files and run the actual task.
        """

        # ── Pass 1: selection ────────────────────────────────────────────
        selection_prompt = (
            f"{_DISCLOSURE_SYSTEM}\n\n"
            f"--- Skill summary ---\n"
            f"{skill.build_summary()}\n\n"
            f"--- User request ---\n"
            f"{args if args else '(no additional input)'}\n\n"
            f"Which files do you need?"
        )

        logger.debug(f"[{skill.name}] Pass 1: requesting sub-file selection")

        router = self._get_agent_router()

        # run_json() must be available on the router.
        # See _AgentRouterJsonMixin for a reference implementation.
        if not hasattr(router, "run_json"):
            logger.warning(
                f"[{skill.name}] AgentRouter lacks run_json(); falling back to full skill load."
            )
            async for chunk in self._execute_full(skill, args):
                yield chunk
            return

        selection: list[str] = await router.run_json(selection_prompt)
        logger.info(f"[{skill.name}] Pass 1 selection: {selection}")

        # Emit a metadata chunk so callers can observe the selection
        yield {
            "type": "skill_disclosure_selection",
            "skill_name": skill.name,
            "selection": selection,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        # ── Pass 2: execution ────────────────────────────────────────────
        async for chunk in self._execute_with_selection(skill, args, selection):
            yield chunk

    # ------------------------------------------------------------------ #
    # Pass 2: load selected files and execute                              #
    # ------------------------------------------------------------------ #

    async def _execute_with_selection(
        self,
        skill: Skill,
        args: str,
        selection: list[str],
    ) -> AsyncIterator[dict]:
        """
        Load only the agent-selected sub-files, build the prompt, and execute.

        Selection semantics
        -------------------
        []              — Agent decided the summary was sufficient; still run
                          the task with the SKILL.md body as minimum context.
        ["__full__"]    — Agent wants the complete SKILL.md body.
        ["a.md", ...]   — Agent selected specific sub-files.

        In all cases the SKILL.md body (core instructions) is always included
        so the agent has the execution context it needs.
        """

        # Always include the core SKILL.md body
        core_prompt = skill.build_prompt(args)
        content_parts: list[str] = [core_prompt]

        if not selection:
            # Summary was enough; proceed with core only
            logger.debug(f"[{skill.name}] Pass 2: no sub-files requested, using core only")

        elif _FULL_SENTINEL in selection:
            # __full__ means the agent already has what it needs from the
            # build_prompt() call above — nothing extra to load.
            logger.debug(f"[{skill.name}] Pass 2: __full__ requested, using core body")

        else:
            # Load the specific sub-files the agent asked for
            valid_selections = [f for f in selection if f in skill.sub_files]
            invalid = set(selection) - set(valid_selections)

            if invalid:
                logger.warning(
                    f"[{skill.name}] Agent requested unknown sub-files: {invalid} — ignoring."
                )

            if valid_selections:
                loaded = skill.load_sub_files(valid_selections)
                logger.info(
                    f"[{skill.name}] Pass 2: loaded {len(loaded)}/{len(valid_selections)} "
                    f"sub-file(s): {list(loaded)}"
                )
                for fname, file_content in loaded.items():
                    header = f"### {fname} ({skill.sub_files.get(fname, '')})"
                    content_parts.append(f"{header}\n\n{file_content}")
            else:
                logger.warning(
                    f"[{skill.name}] No valid sub-files resolved; proceeding with core only."
                )

        combined_content = "\n\n---\n\n".join(content_parts)
        full_prompt = self._wrap_prompt(skill.name, skill.description, combined_content, args)

        logger.debug(
            f"[{skill.name}] Pass 2 prompt: {len(full_prompt)} chars "
            f"(core + {len(content_parts) - 1} sub-file(s))"
        )

        async for chunk in self._get_agent_router().run(full_prompt):
            yield chunk

    # ------------------------------------------------------------------ #
    # Prompt wrapper                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_prompt(
        skill_name: str,
        description: str,
        content: str,
        args: str,
    ) -> str:
        """
        Wrap skill content in the standard execution envelope.

        Kept as a static helper so both _execute_full and
        _execute_with_selection produce identical framing.
        """
        return (
            f'You are executing the "{skill_name}" skill.\n\n'
            f"{description}\n\n"
            f"---\n\n"
            f"{content}\n\n"
            f"---\n\n"
            f"User request: {args if args else '(no additional input)'}"
        )

    # ------------------------------------------------------------------ #
    # Multi-skill progressive disclosure (batch)                          #
    # ------------------------------------------------------------------ #

    async def select_skills(
        self,
        candidate_skills: list[Skill],
        user_request: str,
    ) -> list[str]:
        """
        Show the agent summaries of multiple candidate skills and ask it to
        select which skills to actually execute.

        This is the top-level entry point for multi-skill routing.  Call this
        before execute_skill() when you have several matched skills and want
        the agent to pick the right one(s) rather than running all of them.

        Args:
            candidate_skills: Skills that matched the user's intent (e.g. from
                              SkillLoader.search()).
            user_request:     The user's original message.

        Returns:
            List of skill names the agent wants to run.
        """
        if not candidate_skills:
            return []

        if len(candidate_skills) == 1:
            # No need to ask; just use the one match
            return [candidate_skills[0].name]

        summaries = "\n\n".join(
            f"[{i + 1}] {s.build_summary()}" for i, s in enumerate(candidate_skills)
        )

        selection_prompt = (
            "You are a skill-routing assistant.\n"
            "Given the following available skills and a user request, "
            "return a JSON array of skill NAMES that should be executed.\n"
            "Return ONLY the JSON array — no explanation, no markdown.\n\n"
            f"--- Available skills ---\n{summaries}\n\n"
            f"--- User request ---\n{user_request}\n\n"
            "Which skill name(s) should be executed?"
        )

        router = self._get_agent_router()
        if not hasattr(router, "run_json"):
            logger.warning("AgentRouter lacks run_json(); selecting first skill as fallback.")
            return [candidate_skills[0].name]

        selected_names: list[str] = await router.run_json(selection_prompt)
        valid_names = {s.name for s in candidate_skills}
        result = [n for n in selected_names if n in valid_names]

        if not result:
            logger.warning(
                "Agent returned no valid skill names from multi-skill selection; "
                f"got {selected_names!r}. Defaulting to first candidate."
            )
            return [candidate_skills[0].name]

        return result

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def reset_agent(self) -> None:
        """Reset the agent router (e.g. after settings change)."""
        self._agent_router = None
        logger.info("Agent router reset")

    def list_skills(self) -> list[dict]:
        """
        List all available skills.

        Returns:
            List of skill info dicts (name, description, argument_hint, path,
            sub_files).
        """
        skills = self.skill_loader.get_invocable()
        return [
            {
                "name": s.name,
                "description": s.description,
                "argument_hint": s.argument_hint,
                "path": str(s.path),
                "sub_files": {fname: desc for fname, desc in s.sub_files.items()},
            }
            for s in skills
        ]


# --------------------------------------------------------------------------- #
# Singleton helpers                                                            #
# --------------------------------------------------------------------------- #

_skill_executor: SkillExecutor | None = None


def get_skill_executor() -> SkillExecutor:
    """Get the singleton SkillExecutor instance."""
    global _skill_executor
    if _skill_executor is None:
        _skill_executor = SkillExecutor()
    return _skill_executor
