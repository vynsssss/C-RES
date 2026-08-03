# -*- coding: utf-8 -*-
"""
Wikipedia RAG Tool 

Fetches "Culture of {country}" from Wikipedia, parses into sections,
caches to disk, and returns ALL sections as raw text.

This tool is a pure data reader -- no model calls, no relevance filter.
Coordinator's multi-turn synthesize() feeds all sections into the reading
turn (Turn 1) and lets the model itself extract relevance, with
boundary-aware truncation only if input exceeds the per-model budget.

Flow:
    cultural_helper  ->  wiki_tool.call(country, query)
                            +-> load/fetch data.json        (disk, no model)
                            +-> return {sections_raw: {...}} (all sections)

    coordinator.synthesize()
                        +-> _format_wikipedia(wiki_data, tokenizer, max_tokens)
                                +-> concat all sections, truncate at boundary
                                    if over budget; emit wiki_truncation report
                        +-> _read_evidence(scenario, wiki_block)        <- Turn 1
                                +-> model extracts situation-relevant points
                        +-> generate(synthesis prompt)                  <- Turn 2
"""

import json
import logging
import re
import requests
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level HTML parser ?" defined once, instantiated per _parse_sections call.
# ---------------------------------------------------------------------------
class _SectionParser(HTMLParser):
    """Parse Wikipedia HTML into {section_name: raw_text} dict."""

    def __init__(self):
        super().__init__()
        self.sections: Dict[str, str] = {}
        self.current = "Introduction"
        self.buf: List[str] = []
        self.in_p = False
        self.in_h = False

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_p = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_h = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.in_p = False
            if self.buf:
                self.buf.append("\n\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_h = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self.in_h:
            if self.buf:
                self.sections[self.current] = "".join(self.buf).strip()
            self.current = data
            self.buf = []
        elif self.in_p:
            self.buf.append(data + " ")

    def close(self):
        if self.buf:
            self.sections[self.current] = "".join(self.buf).strip()
        super().close()


class WikipediaRAGToolSummarized:
    """
    Wikipedia RAG Tool 

    Returns all article sections as raw text. Coordinator's reading turn
    extracts relevance during multi-turn synthesis (no retrieval here).

    Backward-compatible name kept so tool_registry.py needs no changes.
    """

    def __init__(
        self,
        corpus_dir: str = "wikipedia_corpus",
        # --- kept for backward compat with tool_registry.py, unused ---
        summarizer_model=None,
        summarizer_tokenizer=None,
        max_summary_chars: int = 600,
        max_total_summary_chars: int = 6000,
        device: str = "cuda",
        cache_file: str = None,
        use_embeddings: bool = False,
        semantic_cache_threshold: float = 0.95,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(exist_ok=True, parents=True)

        # Stats
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Initialized Wikipedia RAG Tool (raw sections, no summarization)")
        logger.info(f"  Corpus: {self.corpus_dir}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "wikipedia_rag"

    @property
    def description(self) -> str:
        return "Wikipedia culture articles - raw sections, coordinator selects relevant ones"

    def call(self, **kwargs) -> Dict[str, Any]:
        """ToolRegistry compatibility entry point."""
        country = kwargs.get("country")
        query = kwargs.get("query", "")

        if not country:
            return {"retrieved": False, "error": "country parameter required"}

        return self.get_culture_article(country, query)

    def get_culture_article(self, country: str, query: str = "") -> Dict[str, Any]:
        """
        Load (or fetch + cache) the "Culture of {country}" Wikipedia article.

        Returns all sections as raw text. Coordinator's reading turn extracts
        situation-relevant points from these sections.

        Args:
            country: Country name.
            query:   Story text (stored for reference, not used for filtering here).

        Returns:
            {
                'retrieved':     True,
                'country':       str,
                'query':         str,
                'sections_raw':  {section_name: full_text, ...},
                'section_names': [str, ...],
                'num_sections':  int,
                'approach':      'raw_sections',
            }
        """
        self.total_queries += 1
        country = country.strip()

        country_key = country.lower().replace(" ", "_")
        country_dir = self.corpus_dir / country_key
        data_file = country_dir / "data.json"

        # --- Load from disk cache ---
        article_data = None
        if data_file.exists():
            try:
                with open(data_file, encoding="utf-8") as f:
                    article_data = json.load(f)
                self.cache_hits += 1
                logger.info(f"[Wikipedia] Cache hit: {country}")
            except Exception as e:
                logger.error(f"[Wikipedia] Cache load failed: {e}")

        # --- Fetch from Wikipedia API if not cached ---
        if article_data is None:
            self.cache_misses += 1
            logger.info(f"[Wikipedia] Fetching: Culture of {country}")
            article_data = self._fetch_and_cache(country, country_dir)

            if not article_data.get("success"):
                return {
                    "retrieved": False,
                    "country": country,
                    "query": query,
                    "sections_raw": {},
                    "section_names": [],
                    "num_sections": 0,
                    "error": article_data.get("error", "Fetch failed"),
                    "approach": "raw_sections",
                }

        sections = article_data.get("sections", {})
        section_names = list(sections.keys())

        logger.info(
            f"[Wikipedia] {country}: {len(sections)} sections, "
            f"{sum(len(v) for v in sections.values()):,} chars total"
        )

        return {
            "retrieved": True,
            "country": country,
            "query": query,
            "sections_raw": sections,        # full text - coordinator filters by relevance
            "section_names": section_names,  # convenience list (same names as sections_raw keys)
            "num_sections": len(sections),
            "approach": "raw_sections",
        }

    # ------------------------------------------------------------------
    # Fetch + cache
    # ------------------------------------------------------------------

    def _fetch_and_cache(self, country: str, country_dir: Path) -> Dict[str, Any]:
        """Fetch article from Wikipedia API and save to disk."""
        article_title = f"Culture of {country}"
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "page": article_title,
                    "format": "json",
                    "prop": "text|sections",
                    "redirects": 1,
                },
                headers={"User-Agent": "CulturalResearchBot/1.0 (Educational research)"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return {"success": False, "error": data["error"].get("info", "API error")}
            if "parse" not in data:
                return {"success": False, "error": "No parse data returned"}

            html_content = data["parse"]["text"].get("*", "")
            sections = self._parse_sections(html_content)

            article_data = {
                "success": True,
                "country": country,
                "title": article_title,
                "sections": sections,
                "total_sections": len(sections),
                "fetched_at": datetime.now().isoformat(),
            }

            # Cache to disk for offline compute nodes
            country_dir.mkdir(parents=True, exist_ok=True)
            with open(country_dir / "data.json", "w", encoding="utf-8") as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[Wikipedia] Cached {country}: {len(sections)} sections")
            return article_data

        except Exception as e:
            logger.exception(f"[Wikipedia] Fetch failed for {country}: {e}")
            return {"success": False, "error": str(e)}


    def _parse_sections(self, html_content: str) -> Dict[str, str]:
        """Parse Wikipedia HTML into {section_name: raw_text} dict."""
        parser = _SectionParser()
        parser.feed(html_content)
        parser.close()


        cleaned = {}
        for name, text in parser.sections.items():
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50:
                cleaned[name] = text

        return cleaned

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / self.total_queries
                if self.total_queries > 0
                else 0.0
            ),
        }