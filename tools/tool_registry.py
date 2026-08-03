# -*- coding: utf-8 -*-
"""
Tool Registry 

Central registry for managing all tools in the C-RES system.
Handles tool registration, calling, and statistics tracking.
- Hofstede Tool       (local CSV)
- Cultural Atlas Tool (local JSON)
- Wikipedia RAG Tool  (hybrid search + caching + unlimited content)
"""

import logging
import os
import traceback
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for all tools used by agents.

    Features:
    - Register tools with unique names
    - Call tools by name with parameters
    - Track usage statistics per tool
    - List available tools
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.call_counts: Dict[str, int] = {}
        logger.info("Tool registry initialized")

    def register(self, name: str, tool: Any, description: Optional[str] = None) -> None:
        """
        Register a tool in the registry.

        Args:
            name:        Unique tool name used when calling via registry.call()
            tool:        Tool instance - must have a .call() method.
            description: Optional description override.
        """
        if name in self.tools:
            logger.warning("Tool '%s' already registered. Overwriting.", name)

        if not hasattr(tool, "call"):
            raise ValueError(
                f"Tool {tool.__class__.__name__} must have a 'call' method"
            )

        self.tools[name] = {
            "tool":        tool,
            "call_method": tool.call,
            "description": description or getattr(tool, "description", "No description available"),
        }
        self.call_counts[name] = 0
        logger.info("[OK] Registered tool: %s", name)

    def call(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool by name with parameters.

        This is the method cultural_helper.py calls.

        Args:
            name:     Tool name (as registered).
            **kwargs: Parameters forwarded to the tool.

        Returns:
            Tool result dict.

        Raises:
            KeyError: If tool is not registered.
        """
        if name not in self.tools:
            raise KeyError(
                f"Tool '{name}' not registered. Available: {list(self.tools.keys())}"
            )

        try:
            result = self.tools[name]["call_method"](**kwargs)
            self.call_counts[name] += 1
            return result

        except (TypeError, AttributeError, NameError) as e:
            # -- Programming bug ------------------------
            # These indicate broken tool code, not missing data.
            # Print to stdout (SLURM .out file) so it's impossible to miss.
            msg = (
                f"\n{'!'*70}\n"
                f"  BUG IN TOOL '{name}': {type(e).__name__}: {e}\n"
                f"  This is a code error, not a data error. Fix the tool.\n"
                f"{'!'*70}"
            )
            print(msg, flush=True)
            logger.error("BUG in tool '%s': %s: %s", name, type(e).__name__, e)
            traceback.print_exc()

            if name == "hofstede_tool":
                return {"success": False, "error": str(e),
                        "error_type": type(e).__name__, "_bug": True}
            else:
                return {"retrieved": False, "error": str(e),
                        "error_type": type(e).__name__, "_bug": True}

        except Exception as e:
            # -- Runtime error (network, file I/O, etc.) ---------------
            logger.error("Error calling tool '%s': %s", name, e)
            traceback.print_exc()

            if name == "hofstede_tool":
                return {"success": False, "error": str(e), "error_type": type(e).__name__}
            else:
                return {"retrieved": False, "error": str(e), "error_type": type(e).__name__}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return info dicts for all registered tools."""
        return [
            {
                "name":        name,
                "description": info["description"],
                "calls":       self.call_counts[name],
            }
            for name, info in self.tools.items()
        ]

    def get_tool(self, name: str) -> Optional[Any]:
        """Return the tool instance, or None if not found."""
        entry = self.tools.get(name)
        return entry["tool"] if entry else None

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    def get_statistics(self) -> Dict[str, Any]:
        total_calls = sum(self.call_counts.values())
        most_used = (
            max(self.call_counts.items(), key=lambda x: x[1])[0]
            if self.call_counts else None
        )
        return {
            "total_tools":  len(self.tools),
            "total_calls":  total_calls,
            "tool_calls":   self.call_counts.copy(),
            "most_used":    most_used,
            "tools":        list(self.tools.keys()),
        }

    def reset_statistics(self) -> None:
        for name in self.call_counts:
            self.call_counts[name] = 0
        logger.info("Tool statistics reset")

    def unregister(self, name: str) -> None:
        if name in self.tools:
            del self.tools[name]
            del self.call_counts[name]
            logger.info("Unregistered tool: %s", name)
        else:
            logger.warning("Tool '%s' not found for unregistration", name)

    def clear(self) -> None:
        self.tools.clear()
        self.call_counts.clear()
        logger.info("Tool registry cleared")

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self.tools)}, total_calls={sum(self.call_counts.values())})"

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.tools


# -- Convenience functions -----------------------------------------------------

def create_registry_with_all_tools(
    hofstede_data_path: Optional[str] = None,
    cultural_atlas_data_path: Optional[str] = None,
) -> ToolRegistry:
    """
    Create a ToolRegistry with all C-RES tools pre-registered.

    Tools:
        1. Hofstede Tool       (local CSV)
        2. Cultural Atlas Tool (local JSON)
        3. Wikipedia RAG Tool  (disk-cached raw sections; all sections fed
                                to coordinator's reading turn -- no retrieval
                                filter, boundary-aware truncation if over
                                budget)

    Args:
        hofstede_data_path:        Path to hofstede_dimensions.csv (uses default if None)
        cultural_atlas_data_path:  Path to cultural_atlas_complete.json (uses default if None)

    Returns:
        ToolRegistry with all 3 tools registered.
    """
    registry = ToolRegistry()

    print("\n" + "=" * 70)
    print("INITIALIZING C-RES TOOL REGISTRY")
    print("=" * 70)

    # -- Tool 1: Hofstede ------------------------------------------------------
    try:
        from hofstede_tool import HofstedeTool
        hofstede_tool = HofstedeTool(data_path=hofstede_data_path)
        registry.register("hofstede_tool", hofstede_tool)
        print(f"[OK] Registered: hofstede_tool")
        if hofstede_data_path:
            print(f"     Data file: {hofstede_data_path}")
    except ImportError as e:
        print(f"[!] Could not import HofstedeTool: {e}")
    except Exception as e:
        print(f"[!] Error initializing Hofstede tool: {e}")

    # -- Tool 2: Cultural Atlas ------------------------------------------------
    try:
        from cultural_atlas_tool import CulturalAtlasTool
        cultural_atlas_tool = CulturalAtlasTool(data_path=cultural_atlas_data_path)
        registry.register("cultural_atlas_tool", cultural_atlas_tool)
        print(f"[OK] Registered: cultural_atlas_tool")
        if cultural_atlas_data_path:
            print(f"     Data file: {cultural_atlas_data_path}")
    except ImportError as e:
        print(f"[!] Could not import CulturalAtlasTool: {e}")
    except Exception as e:
        print(f"[!] Error initializing Cultural Atlas tool: {e}")

    # -- Tool 3: Wikipedia RAG -------------------------------------------------
    try:
        from wikipedia_rag_tool_optimized import WikipediaRAGToolSummarized
        wikipedia_tool = WikipediaRAGToolSummarized(
            corpus_dir="wikipedia_corpus",
        )
        registry.register("wikipedia_rag", wikipedia_tool)
        print(f"[OK] Registered: wikipedia_rag")
        print(f"     Corpus directory: wikipedia_corpus/")
        print(f"     All sections fed raw to coordinator (no retrieval filter)")
    except ImportError as e:
        print(f"[!] Could not import WikipediaRAGToolSummarized: {e}")
    except Exception as e:
        print(f"[!] Error initializing Wikipedia RAG tool: {e}")

    # -- Summary ---------------------------------------------------------------
    n = len(registry)
    print("\n" + "=" * 70)
    print(f"TOOL REGISTRY COMPLETE - {n} tool{'s' if n != 1 else ''} registered")
    for t in registry.list_tools():
        print(f"  - {t['name']}")
    print("=" * 70 + "\n")

    return registry


def verify_tools(registry: ToolRegistry) -> Dict[str, bool]:
    """Verify all required C-RES tools are available."""
    required = ["hofstede_tool", "cultural_atlas_tool", "wikipedia_rag"]
    return {name: registry.has_tool(name) for name in required}


def print_tool_info(registry: ToolRegistry) -> None:
    """
    Print information about all registered tools.
    """
    print("\n" + "=" * 70)
    print("REGISTERED TOOLS")
    print("=" * 70)

    for tool_dict in registry.list_tools():   # single call - O(n)
        name = tool_dict["name"]
        print(f"\n{name}:")
        print(f"  Description: {tool_dict['description']}")
        print(f"  Calls: {tool_dict['calls']}")

        tool = registry.get_tool(name)
        if tool and hasattr(tool, "TOOL_INFO"):
            info = tool.TOOL_INFO
            if "parameters" in info:
                print(f"  Parameters: {info['parameters']}")

    print("\n" + "=" * 70)


# -- Tests ---------------------------------------------------------------------

def test_tool_registry() -> None:
    print("Testing Tool Registry - C-RES-6 Version")
    print("=" * 70)

    class MockTool:
        @property
        def description(self):
            return "Mock tool for testing"
        def call(self, **kwargs):
            return {"success": True, "data": kwargs}

    registry = ToolRegistry()
    registry.register("mock_tool", MockTool())
    assert len(registry) == 1

    result = registry.call("mock_tool", test_param="value")
    assert result["success"] is True
    assert result["data"]["test_param"] == "value"

    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "mock_tool"

    stats = registry.get_statistics()
    assert stats["total_tools"] == 1
    assert stats["total_calls"] == 1
    assert stats["most_used"] == "mock_tool"

    try:
        registry.call("nonexistent_tool")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    assert registry.has_tool("mock_tool") is True
    assert registry.has_tool("nonexistent") is False

    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    test_tool_registry()