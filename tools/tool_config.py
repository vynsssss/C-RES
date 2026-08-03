# -*- coding: utf-8 -*-
"""
Tool Configuration 
"""

from typing import Dict, Any, List

# -- Tool activation flags -----------------------------------------------------
ACTIVE_TOOLS = {
    "hofstede":       True,    # Phase 1
    "cultural_atlas": True,    # Phase 2
    "wikipedia_rag":  True,    # Phase 3 
}

# -- Stub responses ------------------------------------------------------------
# Returned when a tool is disabled.  Each tool has its own response format:
#   hofstede       -> {"success": False, ...}
#   cultural_atlas -> {"retrieved": False, ...}
#   wikipedia_rag  -> {"retrieved": False, ...}
#
# Pre-built dict gives O(1) lookup instead of an if/elif chain.
_STUB_RESPONSES: Dict[str, Dict[str, Any]] = {}  # populated by _build_stubs() below


def _build_stubs() -> Dict[str, Dict[str, Any]]:
    """Build stub response dict once at module load."""
    tools = {
        "hofstede":       {"success":   False},
        "cultural_atlas": {"retrieved": False},
        "wikipedia_rag":  {"retrieved": False},
    }
    return {
        name: {**base, "status": "disabled", "tool": name,
               "message": f"{name} is currently disabled for incremental testing"}
        for name, base in tools.items()
    }


_STUB_RESPONSES: Dict[str, Dict[str, Any]] = _build_stubs()

_GENERIC_STUB_BASE: Dict[str, Any] = {
    "status": "disabled",
    "data":   None,
}


def get_stub_response(tool_name: str) -> Dict[str, Any]:
    """
    Return empty response for a disabled tool.

    Uses a pre-built dict lookup - O(1) instead of the old if/elif chain.
    Falls back to a generic stub for unknown tool names.
    """
    if tool_name in _STUB_RESPONSES:
        return dict(_STUB_RESPONSES[tool_name])   # return a copy so callers can't mutate the template
    # Generic stub for unknown tools
    return {
        **_GENERIC_STUB_BASE,
        "tool":    tool_name,
        "message": f"{tool_name} is currently disabled for incremental testing",
    }


# -- Helper functions ----------------------------------------------------------

def is_tool_enabled(tool_name: str) -> bool:
    """Check if a tool is currently enabled."""
    return ACTIVE_TOOLS.get(tool_name, False)


def get_active_tools() -> List[str]:
    """Return names of currently active tools."""
    return [name for name, active in ACTIVE_TOOLS.items() if active]


def get_tool_status() -> Dict[str, bool]:
    """Return copy of the current tool activation state."""
    return ACTIVE_TOOLS.copy()


# -- Configuration presets -----------------------------------------------------
PRESETS: Dict[str, Dict[str, bool]] = {
    # C-RES progressive testing
    "phase1_hofstede_only":    {"hofstede": True,  "cultural_atlas": False, "wikipedia_rag": False},
    "phase2_hofstede_atlas":   {"hofstede": True,  "cultural_atlas": True,  "wikipedia_rag": False},
    "phase3_all_tools":        {"hofstede": True,  "cultural_atlas": True,  "wikipedia_rag": True},
    # Individual tool testing
    "test_hofstede_only":      {"hofstede": True,  "cultural_atlas": False, "wikipedia_rag": False},
    "test_atlas_only":         {"hofstede": False, "cultural_atlas": True,  "wikipedia_rag": False},
    "test_wikipedia_only":     {"hofstede": False, "cultural_atlas": False, "wikipedia_rag": True},
    # Pair testing
    "test_hofstede_wikipedia": {"hofstede": True,  "cultural_atlas": False, "wikipedia_rag": True},
    "test_atlas_wikipedia":    {"hofstede": False, "cultural_atlas": True,  "wikipedia_rag": True},
    # Disable all (baseline)
    "baseline_no_tools":       {"hofstede": False, "cultural_atlas": False, "wikipedia_rag": False},
}


def load_preset(preset_name: str) -> None:
    """Load a configuration preset."""
    global ACTIVE_TOOLS
    if preset_name in PRESETS:
        ACTIVE_TOOLS = PRESETS[preset_name].copy()
        print(f"Loaded preset: {preset_name}")
        print(f"Active tools:  {get_active_tools()}")
    else:
        print(f"Unknown preset: {preset_name}")
        print(f"Available:      {list(PRESETS.keys())}")


# -- Validation ----------------------------------------------------------------

def validate_config() -> bool:
    """Validate that tool names match what the codebase expects."""
    expected = {"hofstede", "cultural_atlas", "wikipedia_rag"}
    actual   = set(ACTIVE_TOOLS.keys())

    if expected != actual:
        print(f"\nWARNING: Tool configuration mismatch!")
        print(f"  Expected: {sorted(expected)}")
        print(f"  Actual:   {sorted(actual)}")
        print(f"  Missing:  {sorted(expected - actual)}")
        print(f"  Extra:    {sorted(actual - expected)}")
        return False

    print("Tool configuration validated")
    return True


# -- Tool metadata (for display / reporting) -----------------------------------
TOOL_INFO: Dict[str, Dict[str, Any]] = {
    "hofstede": {
        "name":           "Hofstede Cultural Dimensions",
        "description":    "6 cultural dimensions from CSV",
        "source":         "hofstede_dimensions.csv",
        "response_field": "success",
    },
    "cultural_atlas": {
        "name":           "Cultural Atlas Database",
        "description":    "Detailed cultural practices from local JSON",
        "source":         "cultural_atlas_complete.json",
        "response_field": "retrieved",
    },
    "wikipedia_rag": {
        "name":           "Wikipedia RAG",
        "description":    "Fetches and disk-caches 'Culture of {country}' articles as raw sections",
        "source":         "wikipedia_corpus/",
        "response_field": "retrieved",
        "features": [
            "Disk cache per country (wikipedia_corpus/<country>/data.json)",
            "Returns all sections as raw text -- no model calls in the tool",
            "Coordinator feeds ALL sections to the reading turn (no retrieval filter)",
            "Boundary-aware truncation if over input budget - drops trailing sections",
        ],
    },
}


def print_tool_info() -> None:
    """Print information about all available tools."""
    print("\n" + "=" * 70)
    print("C-RES-6 TOOLS")
    print("=" * 70)
    for tool_id, info in TOOL_INFO.items():
        status = "ENABLED" if is_tool_enabled(tool_id) else "DISABLED"
        print(f"\n{info['name']} ({tool_id}): {status}")
        print(f"  Description: {info['description']}")
        print(f"  Source:      {info['source']}")
        if "features" in info:
            print("  Features:")
            for feature in info["features"]:
                print(f"    {feature}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("C-RES-6 Tool Configuration")
    print("=" * 70)
    validate_config()
    print(f"\nCurrent configuration:")
    for tool, enabled in ACTIVE_TOOLS.items():
        print(f"  {tool:20s}  {'ENABLED' if enabled else 'DISABLED'}")
    print(f"\nAvailable presets:")
    for name in PRESETS:
        print(f"  {name}")
    print_tool_info()