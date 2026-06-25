"""
Medical info search tool — searches MedlinePlus for patient-friendly health information.

Uses the MedlinePlus Web Service (free, no API key required).
API docs: https://medlineplus.gov/about/developers/webservices/

Note: Also includes PubMed academic search in the same file.
"""

from __future__ import annotations

import logging
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 10) -> bytes:
    """Fetch URL with proper User-Agent header."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VitalAgent/1.0 (research project)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# MedlinePlus: Patient-friendly medical information
# ---------------------------------------------------------------------------

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"


def _medlineplus_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search MedlinePlus health topics and return titles + summaries."""
    params = {
        "db": "healthTopics",
        "term": query,
        "retmax": str(max_results),
        "rettype": "brief",
    }
    url = f"{MEDLINEPLUS_URL}?{urllib.parse.urlencode(params)}"
    xml_data = _fetch_url(url).decode()

    root = ET.fromstring(xml_data)
    articles = []

    for doc in root.findall(".//document"):
        entry: dict[str, str] = {}
        doc_url = doc.get("url", "")
        if doc_url:
            entry["url"] = doc_url

        for content in doc.findall("content"):
            name = content.get("name", "")
            text = content.text or ""
            text = _strip_html(text).strip()

            if name == "title" and text:
                entry["title"] = text
            elif name == "FullSummary" and text:
                # Truncate long summaries
                if len(text) > 1500:
                    text = text[:1500] + "..."
                entry["summary"] = text
            elif name == "snippet" and text and "summary" not in entry:
                entry["snippet"] = text

        if entry.get("title"):
            articles.append(entry)

    return articles


@register_tool(
    name="medical_info_search",
    description=(
        "Search MedlinePlus for patient-friendly health information about "
        "cardiac conditions, ECG findings, symptoms, and medical terms. "
        "Returns plain-language summaries from the U.S. National Library of Medicine. "
        "Use when the user asks 'what is ...', 'explain ...', or needs "
        "accessible medical information rather than academic literature."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Medical topic to search (e.g. 'atrial fibrillation', 'heart rate variability', 'bradycardia').",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Default 3.",
            "required": False,
            "default": 3,
        },
    },
)
def medical_info_search(
    query: str,
    max_results: int = 3,
) -> dict[str, Any]:
    """Search MedlinePlus for patient-friendly medical information."""

    if not query.strip():
        return {"success": False, "error": "Empty query."}

    try:
        articles = _medlineplus_search(query, max_results=max_results)
    except Exception as e:
        logger.error("MedlinePlus search failed: %s", e)
        return {"success": False, "error": f"MedlinePlus search failed: {e}"}

    if not articles:
        return {
            "success": True,
            "data": {
                "articles": [],
                "note": "No results found. Try broader or different search terms.",
            },
        }

    return {
        "success": True,
        "data": {
            "query": query,
            "source": "MedlinePlus (U.S. National Library of Medicine)",
            "num_results": len(articles),
            "articles": articles,
        },
    }


# ---------------------------------------------------------------------------
# PubMed: Academic medical literature
# ---------------------------------------------------------------------------

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _esearch(query: str, max_results: int = 3) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "sort": "relevance",
        "retmode": "json",
    }
    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    data = json.loads(_fetch_url(url).decode())
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch_abstracts(pmids: list[str]) -> list[dict[str, str]]:
    """Fetch title and abstract for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"
    xml_data = _fetch_url(url, timeout=15).decode()

    root = ET.fromstring(xml_data)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None and title_el.text else "No title"

        abstract_parts = []
        for abs_text in article.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = (
            " ".join(abstract_parts) if abstract_parts else "No abstract available."
        )

        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else None

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        entry: dict[str, str] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
        }
        if year:
            entry["year"] = year
        if journal:
            entry["journal"] = journal

        articles.append(entry)

    return articles


@register_tool(
    name="medical_knowledge",
    description=(
        "Search PubMed academic literature for evidence-based information about "
        "cardiac conditions, ECG findings, treatments, or medical research. "
        "Returns article titles and abstracts from peer-reviewed journals. "
        "Use when the user needs clinical evidence, research findings, or "
        "detailed medical information beyond patient-level summaries."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Medical search query (e.g. 'atrial fibrillation treatment guidelines', 'low HRV clinical significance').",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of articles to return. Default 3.",
            "required": False,
            "default": 3,
        },
    },
)
def medical_knowledge(
    query: str,
    max_results: int = 3,
) -> dict[str, Any]:
    """Search PubMed for academic medical literature."""

    if not query.strip():
        return {"success": False, "error": "Empty query."}

    try:
        pmids = _esearch(query, max_results=max_results)
    except Exception as e:
        logger.error("PubMed search failed: %s", e)
        return {"success": False, "error": f"PubMed search failed: {e}"}

    if not pmids:
        return {
            "success": True,
            "data": {
                "articles": [],
                "note": "No results found. Try broader or different search terms.",
            },
        }

    try:
        articles = _efetch_abstracts(pmids)
    except Exception as e:
        logger.error("PubMed fetch failed: %s", e)
        return {"success": False, "error": f"Failed to fetch article details: {e}"}

    return {
        "success": True,
        "data": {
            "query": query,
            "source": "PubMed (NCBI)",
            "num_results": len(articles),
            "articles": articles,
        },
    }
