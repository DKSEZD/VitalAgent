from __future__ import annotations

import pytest

from agent.proactive.guidelines import GuidelineLibrary


def test_list_guideline_ids_returns_all_three() -> None:
    lib = GuidelineLibrary()

    assert set(lib.list_guideline_ids()) == {"af-2023", "brady-2018", "svt-2015"}


def test_get_summary_text_returns_non_empty_markdown_for_each_guideline() -> None:
    lib = GuidelineLibrary()

    for guideline_id in lib.list_guideline_ids():
        summary = lib.get_summary_text(guideline_id)
        assert len(summary) > 500


def test_parser_extracts_sections_from_each_guideline() -> None:
    lib = GuidelineLibrary()

    assert len(lib.list_sections("af-2023")) >= 6
    assert len(lib.list_sections("brady-2018")) >= 5
    assert len(lib.list_sections("svt-2015")) >= 6


def test_specific_sections_are_retrievable() -> None:
    lib = GuidelineLibrary()

    af = lib.get_section("af-2023", "ahre-24h-plus")
    brady = lib.get_section("brady-2018", "nocturnal-bradycardia")
    svt = lib.get_section("svt-2015", "tachycardia-definition")

    assert af is not None
    assert "24 hours" in af.title.lower()

    assert brady is not None
    assert (
        "nocturnal" in brady.content.lower()
        or "asleep" in brady.content.lower()
        or "sleep" in brady.content.lower()
    )

    assert svt is not None
    assert "100" in svt.content


def test_get_section_returns_none_for_unknown_section() -> None:
    lib = GuidelineLibrary()

    assert lib.get_section("af-2023", "does-not-exist") is None


def test_get_summary_text_raises_for_unknown_guideline() -> None:
    lib = GuidelineLibrary()

    with pytest.raises(FileNotFoundError):
        lib.get_summary_text("unknown-guideline")


def test_section_ids_are_unique_within_each_guideline() -> None:
    lib = GuidelineLibrary()

    for guideline_id in lib.list_guideline_ids():
        section_ids = [section.section_id for section in lib.list_sections(guideline_id)]
        assert len(section_ids) == len(set(section_ids))


def test_section_guideline_id_matches_parent() -> None:
    lib = GuidelineLibrary()

    for guideline_id in lib.list_guideline_ids():
        for section in lib.list_sections(guideline_id):
            assert section.guideline_id == guideline_id
