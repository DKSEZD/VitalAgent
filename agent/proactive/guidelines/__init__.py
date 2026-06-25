"""
Guideline library — indexed access to clinical-guideline content.

Two-layer storage:
    summaries/  — executive summaries, one markdown file per guideline,
                  structured as ``## [guideline-id.section-id] title``
                  so sections can be individually fetched.
    fulltext/   — optional full-text content (per-section markdown or
                  per-guideline PDF extracts). Not required for Phase 1;
                  the LLM can request a section and fall back to summary
                  when full text is absent.

The library is intentionally a thin wrapper around the filesystem. It
does not perform embedding / RAG; the LLM either has the summary in its
system prompt or calls the read_guideline_section tool for a specific
section by ID. Guideline content changes rarely, so caching at module
import time is acceptable.

Design note: this module lives under ``agent/proactive/`` for Phase 1.
If Reactive or other components later need guideline access, the entire
``guidelines`` directory and ``GuidelineLibrary`` class can be relocated
to a shared location (e.g. ``agent/shared/``) — callers depend only on
the public API, not the file path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Default location: same directory as this file.
_GUIDELINES_DIR = Path(__file__).parent
_SUMMARY_DIR = _GUIDELINES_DIR / "summaries"
_FULLTEXT_DIR = _GUIDELINES_DIR / "fulltext"

# Section header pattern: ``## [guideline-id.section-id] title``
_SECTION_HEADER_RE = re.compile(r"^##\s+\[([a-z0-9\-]+)\.([a-z0-9\-]+)\]\s+(.*)$")


@dataclass(frozen=True)
class GuidelineSection:
    """A single addressable section within a guideline summary."""

    guideline_id: str
    section_id: str
    title: str
    content: str  # plain markdown, no header line

    @property
    def fully_qualified_id(self) -> str:
        return f"{self.guideline_id}.{self.section_id}"


@dataclass(frozen=True)
class Guideline:
    """A guideline: header metadata plus an ordered list of sections."""

    guideline_id: str
    header: str  # everything before the first ``## [...]`` header
    sections: tuple[GuidelineSection, ...]

    def section(self, section_id: str) -> GuidelineSection | None:
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None


class GuidelineLibrary:
    """
    Filesystem-backed store of guideline summaries and sections.

    Typical usage:
        lib = GuidelineLibrary()
        summary = lib.get_summary_text("af-2023")
        section = lib.get_section("af-2023", "ahre-24h-plus")
        all_summaries = lib.list_all_summaries()
    """

    def __init__(
        self,
        summary_dir: Path | None = None,
        fulltext_dir: Path | None = None,
    ) -> None:
        self._summary_dir = summary_dir or _SUMMARY_DIR
        self._fulltext_dir = fulltext_dir or _FULLTEXT_DIR
        if not self._summary_dir.exists():
            logger.warning(
                "Guideline summary directory not found: %s", self._summary_dir
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_guideline_ids(self) -> list[str]:
        """Return the IDs of all known guidelines (derived from filenames)."""
        if not self._summary_dir.exists():
            return []
        return sorted(p.stem.replace("_", "-") for p in self._summary_dir.glob("*.md"))

    # ------------------------------------------------------------------
    # Full-summary access
    # ------------------------------------------------------------------

    def get_summary_text(self, guideline_id: str) -> str:
        """
        Return the full markdown text of a guideline's executive summary.
        Suitable for injection into an LLM system prompt.
        """
        path = self._summary_path(guideline_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Guideline summary not found: {guideline_id} (expected at {path})"
            )
        return path.read_text(encoding="utf-8")

    def list_all_summaries(self) -> dict[str, str]:
        """Return ``{guideline_id: summary_markdown}`` for every guideline."""
        return {gid: self.get_summary_text(gid) for gid in self.list_guideline_ids()}

    # ------------------------------------------------------------------
    # Section-level access
    # ------------------------------------------------------------------

    def get_guideline(self, guideline_id: str) -> Guideline:
        """Parse and return the structured form of a guideline summary."""
        return _parse_guideline_cached(self._summary_path(guideline_id), guideline_id)

    def get_section(
        self,
        guideline_id: str,
        section_id: str,
    ) -> GuidelineSection | None:
        """Look up one section by ID. Returns None if not found."""
        guideline = self.get_guideline(guideline_id)
        return guideline.section(section_id)

    def list_sections(self, guideline_id: str) -> list[GuidelineSection]:
        """Return all sections for a guideline in document order."""
        return list(self.get_guideline(guideline_id).sections)

    # ------------------------------------------------------------------
    # Full-text access (Phase 2 — currently only checks for availability)
    # ------------------------------------------------------------------

    def has_fulltext(self, guideline_id: str, section_id: str) -> bool:
        """Return whether a full-text expansion is available for a section."""
        path = self._fulltext_path(guideline_id, section_id)
        return path.exists()

    def get_fulltext(
        self,
        guideline_id: str,
        section_id: str,
    ) -> str | None:
        """
        Return the full-text expansion for a section, if available.
        Returns None if the section has no full-text file (fall back to
        summary content).
        """
        path = self._fulltext_path(guideline_id, section_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summary_path(self, guideline_id: str) -> Path:
        filename = guideline_id.replace("-", "_") + ".md"
        return self._summary_dir / filename

    def _fulltext_path(self, guideline_id: str, section_id: str) -> Path:
        folder = self._fulltext_dir / guideline_id.replace("-", "_")
        return folder / f"{section_id}.md"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _parse_guideline_cached(path: Path, guideline_id: str) -> Guideline:
    """Parse a guideline summary into its structured form, cached by path."""
    if not path.exists():
        raise FileNotFoundError(
            f"Guideline summary not found: {guideline_id} (expected at {path})"
        )
    text = path.read_text(encoding="utf-8")
    return _parse_guideline(text, guideline_id)


def _parse_guideline(text: str, expected_guideline_id: str) -> Guideline:
    """Parse markdown text into Guideline(header, sections)."""
    lines = text.splitlines()
    header_lines: list[str] = []
    sections: list[GuidelineSection] = []

    # State while walking lines
    in_header = True
    current_gid: str | None = None
    current_sid: str | None = None
    current_title: str | None = None
    current_body: list[str] = []

    def _flush_section() -> None:
        if current_gid is None or current_sid is None or current_title is None:
            return
        sections.append(
            GuidelineSection(
                guideline_id=current_gid,
                section_id=current_sid,
                title=current_title,
                content="\n".join(current_body).strip(),
            )
        )

    for line in lines:
        match = _SECTION_HEADER_RE.match(line)
        if match:
            gid, sid, title = match.group(1), match.group(2), match.group(3).strip()
            if gid != expected_guideline_id:
                logger.warning(
                    "Section header declares guideline '%s' but file is for '%s' (section '%s')",
                    gid,
                    expected_guideline_id,
                    sid,
                )
            _flush_section()
            in_header = False
            current_gid, current_sid, current_title = gid, sid, title
            current_body = []
            continue

        if in_header:
            header_lines.append(line)
        else:
            current_body.append(line)

    _flush_section()

    return Guideline(
        guideline_id=expected_guideline_id,
        header="\n".join(header_lines).strip(),
        sections=tuple(sections),
    )


# ---------------------------------------------------------------------------
# Convenience: module-level singleton
# ---------------------------------------------------------------------------

_library_singleton: GuidelineLibrary | None = None


def get_library() -> GuidelineLibrary:
    """Get or lazily initialise the default library instance."""
    global _library_singleton
    if _library_singleton is None:
        _library_singleton = GuidelineLibrary()
    return _library_singleton


def iter_all_sections() -> Iterable[GuidelineSection]:
    """Yield every section across every guideline (for debugging / listing)."""
    lib = get_library()
    for gid in lib.list_guideline_ids():
        yield from lib.list_sections(gid)