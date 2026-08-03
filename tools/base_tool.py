# -*- coding: utf-8 -*-
"""
Base Tool Class

Tools provide information retrieval capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """
  
    All tools must implement:
    - name:        Tool identifier
    - description: What the tool does
    - call:        Main execution method
    """

    def __init__(self):
        self.usage_count: int = 0
        # NOTE: call_history was removed.
        # The old code appended a dict to call_history on every tool call
        # (up to 7,899 entries per run at 432 bytes each ≈ 10 MB total across
        # 3 tools) but the list was never read anywhere in the codebase.
        # usage_count is sufficient for statistics.

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name / identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for agents."""
        pass

    @abstractmethod
    def call(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            Dict with tool results.
        """
        pass

    def _log_call(self, *args, **kwargs) -> None:
        """
        Increment usage counter.  Called by subclasses after each tool call.
        """
        self.usage_count += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return tool usage statistics."""
        return {
            "name":        self.name,
            "total_calls": self.usage_count,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_count = 0