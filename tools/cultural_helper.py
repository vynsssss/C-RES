# -*- coding: utf-8 -*-
"""
Cultural Helper — Calls tools, returns raw data.

Tool configurations (for ablation studies):
- 'hofstede':       Hofstede only
- 'atlas':          Cultural Atlas only
- 'wiki':           Wikipedia only
- 'hofstede_atlas': Hofstede + Cultural Atlas
- 'hofstede_wiki':  Hofstede + Wikipedia
- 'atlas_wiki':     Cultural Atlas + Wikipedia
- 'all':            All 3 tools (default)
"""

import logging
from typing import Dict, Any, Optional

from tool_config import is_tool_enabled, get_stub_response

logger = logging.getLogger(__name__)

VALID_CONFIGS = {
    "hofstede", "atlas", "wiki",
    "hofstede_atlas", "hofstede_wiki", "atlas_wiki",
    "all",
}

# Which tools each config enables
CONFIG_TOOLS = {
    "hofstede":       {"hofstede"},
    "atlas":          {"cultural_atlas"},
    "wiki":           {"wikipedia_rag"},
    "hofstede_atlas": {"hofstede", "cultural_atlas"},
    "hofstede_wiki":  {"hofstede", "wikipedia_rag"},
    "atlas_wiki":     {"cultural_atlas", "wikipedia_rag"},
    "all":            {"hofstede", "cultural_atlas", "wikipedia_rag"},
}


class CulturalHelper:
    """
    Calls cultural tools and returns raw results.

    Usage:
        helper = CulturalHelper(tool_registry, tool_config='all')
        result = helper.get_evidence(sample)
        # result = {
        #     'hofstede_data':       {...} or None,
        #     'cultural_atlas_data': {...} or None,
        #     'wikipedia_data':      {...} or None,
        #     'tools_used':          ['hofstede_tool', 'cultural_atlas_tool', ...],
        # }

    Then pass to Coordinator.synthesize():
        coordinator.synthesize(
            baseline_reasoning=...,
            baseline_answer=...,
            hofstede_data=result['hofstede_data'],
            cultural_atlas_data=result['cultural_atlas_data'],
            wikipedia_data=result['wikipedia_data'],
        )
    """

    def __init__(self, tool_registry, tool_config: str = "all"):
        self.tool_registry = tool_registry
        self.tool_config = tool_config if tool_config in VALID_CONFIGS else "all"

        # Stats
        self.hofstede_calls = 0
        self.cultural_atlas_calls = 0
        self.wikipedia_calls = 0
        self.total_calls = 0

        logger.info(f"CulturalHelper: config={self.tool_config}")

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def get_evidence(
        self,
        sample: Dict[str, Any],
        tool_config: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call enabled tools for this sample's country.

        Args:
            sample: Normalized NorMAD sample (needs 'country', 'story').
            tool_config: Override default config for this call.

        Returns:
            Dict with raw tool results + tools_used list.
        """
        config = tool_config if tool_config in VALID_CONFIGS else self.tool_config
        enabled = CONFIG_TOOLS[config]

        country = self._normalize_country(sample.get("country", ""))
        story = sample.get("story", "")

        hofstede_data = None
        cultural_atlas_data = None
        wikipedia_data = None
        tools_used = []

        # Hofstede
        if "hofstede" in enabled:
            if is_tool_enabled("hofstede"):
                hofstede_data = self._call_tool(
                    "hofstede_tool", country=country
                )
                if hofstede_data and hofstede_data.get("success"):
                    tools_used.append("hofstede_tool")
                    self.hofstede_calls += 1
            else:
                hofstede_data = get_stub_response("hofstede")

        # Cultural Atlas
        if "cultural_atlas" in enabled:
            if is_tool_enabled("cultural_atlas"):
                cultural_atlas_data = self._call_tool(
                    "cultural_atlas_tool", country=country, query=story
                )
                if cultural_atlas_data and cultural_atlas_data.get("retrieved"):
                    tools_used.append("cultural_atlas_tool")
                    self.cultural_atlas_calls += 1
            else:
                cultural_atlas_data = get_stub_response("cultural_atlas")

        # Wikipedia
        if "wikipedia_rag" in enabled:
            if is_tool_enabled("wikipedia_rag"):
                wikipedia_data = self._call_tool(
                    "wikipedia_rag", country=country, query=story
                )
                if wikipedia_data and wikipedia_data.get("retrieved"):
                    tools_used.append("wikipedia_rag")
                    self.wikipedia_calls += 1
            else:
                wikipedia_data = get_stub_response("wikipedia_rag")

        self.total_calls += 1

        return {
            "hofstede_data": hofstede_data,
            "cultural_atlas_data": cultural_atlas_data,
            "wikipedia_data": wikipedia_data,
            "tools_used": tools_used,
            "tool_config": config,
        }

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------
    def _call_tool(self, tool_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Call a tool via registry with error handling."""
        try:
            return self.tool_registry.call(tool_name, **kwargs)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Country normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_country(country: str) -> str:
        """Normalize country name for tool lookup."""
        if not country:
            return country

        special = {
            "usa": "United States",
            "us": "United States",
            "united_states_of_america": "United States",
            "uk": "United Kingdom",
            "uae": "United Arab Emirates",
            "drc": "Democratic Republic of Congo",
            "south_korea": "South Korea",
            "north_korea": "North Korea",
        }

        lower = country.lower().strip()
        if lower in special:
            return special[lower]

        # Title case with lowercase articles
        normalized = lower.replace("_", " ").replace("-", " ").title()
        lowercase_words = {"of", "the", "and", "de", "la", "le"}
        words = normalized.split()
        return " ".join(
            w.lower() if i > 0 and w.lower() in lowercase_words else w
            for i, w in enumerate(words)
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": "CulturalHelper",
            "tool_config": self.tool_config,
            "total_calls": self.total_calls,
            "hofstede_calls": self.hofstede_calls,
            "cultural_atlas_calls": self.cultural_atlas_calls,
            "wikipedia_calls": self.wikipedia_calls,
            "avg_tools_per_call": (
                (self.hofstede_calls + self.cultural_atlas_calls + self.wikipedia_calls)
                / self.total_calls
                if self.total_calls > 0
                else 0
            ),
        }