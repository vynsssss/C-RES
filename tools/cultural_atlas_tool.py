# -*- coding: utf-8 -*-
"""
Cultural Atlas Tool  

Retrieves rich cultural information from Cultural Atlas database.
Provides detailed cultural practices, values, etiquette, and norms.

FEATURES:
- Class-level data caching (load JSON once, share across instances)
- Comprehensive country name normalization
- Returns raw axes data (let LLM do summarization)
- NOT FOUND tracking (clean error messages)
- Efficient lookups
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
import logging

from base_tool import BaseTool

logger = logging.getLogger(__name__)


class CulturalAtlasTool(BaseTool):
    """
    Tool for querying Cultural Atlas cultural information.
    """
    
    # Class-level cache (shared across all instances)
    _data_cache = None
    _cache_path = None
    
    def __init__(self, data_path: Optional[str] = None):
        super().__init__()
        
        # Determine data path
        if data_path is None:
            data_path = Path(__file__).parent / "cultural_atlas_complete.json"
        else:
            data_path = Path(data_path)
        
        # Validate file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Cultural Atlas data file not found: {data_path}")
        
        # Load from cache or read from file
        if (CulturalAtlasTool._data_cache is None or 
            CulturalAtlasTool._cache_path != str(data_path)):
            logger.info(f"Loading Cultural Atlas data from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Cache for future instances
            CulturalAtlasTool._data_cache = self.data.copy()
            CulturalAtlasTool._cache_path = str(data_path)
            logger.info(f"Cached Cultural Atlas data ({len(self.data)} countries)")
        else:
            logger.debug("Using cached Cultural Atlas data")
            self.data = CulturalAtlasTool._data_cache.copy()
        
        # Track countries not found
        self.not_found_countries: Set[str] = set()
    
    @property
    def name(self) -> str:
        return "cultural_atlas_tool"
    
    @property
    def description(self) -> str:
        return (
            "Query Cultural Atlas for detailed cultural information about a country. "
            "Returns core values, family structure, etiquette, communication styles, "
            "and other cultural practices."
        )
    
    # Country name mapping for common variants
    COUNTRY_ALIASES = {
        # Americas
        'united states': 'united states of america',
        'usa': 'united states of america',
        'us': 'united states of america',
        'america': 'united states of america',
        
        # Europe
        'england': 'united kingdom',
        'britain': 'united kingdom',
        'great britain': 'united kingdom',
        'uk': 'united kingdom',
        
        # Asia
        'korea': 'south korea',
        'republic of korea': 'south korea',
        
        # Middle East
        'uae': 'united arab emirates',
        'emirates': 'united arab emirates',
        
        # Africa
        'rsa': 'south africa',
        
        # Hyphen / accent normalization
        # cultural_helper replaces hyphens with spaces before calling this tool,
        # but the Atlas JSON key keeps the hyphen: 'timor-leste'
        'timor leste': 'timor-leste',
        # cultural_helper title-cases 'türkiye' → 'Türkiye', lowered here to 'türkiye'
        # Atlas JSON key is 'türkiye' (with ü) so this is a no-op confirmation,
        # but the ASCII variant needs an explicit mapping.
        'turkiye': 'türkiye',
    }
    
    def _normalize_country(self, country: str) -> str:
        """
        Normalize country name for lookup.
        
        Handles:
        - Case: "EGYPT" → "egypt"
        - Underscores: "united_states" → "united states"
        - Hyphens: "south-korea" → "south korea"
        - Extra spaces
        - Common name variants
        
        Args:
            country: Country name
            
        Returns:
            Normalized country name
        """
        normalized = country.lower().strip()
        normalized = normalized.replace('_', ' ')
        normalized = normalized.replace('-', ' ')
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        # Check for known aliases
        if normalized in self.COUNTRY_ALIASES:
            normalized = self.COUNTRY_ALIASES[normalized]
        
        return normalized
    
    def call(self, country: str, **_kwargs) -> Dict[str, Any]:
        """
        Get Cultural Atlas information for a country.
        
        Returns RAW axes data - let the LLM do interpretation and summarization.
        
        Args:
            country: Country name (case-insensitive)
        
        Returns:
            Dict with cultural information:
            {
                'retrieved': True,
                'country': 'Egypt',
                'country_normalized': 'egypt',
                'url': 'https://culturalatlas.sbs.com.au/egyptian-culture',
                'axes': {
                    'Core Concepts': {
                        'content': {
                            'core_values': ['Honour', 'Loyalty', ...],
                            'desc': '...',
                            ...
                        },
                        'url': '...'
                    },
                    'Family': {...},
                    'Etiquette': {...},
                    ...
                },
                'summary': 'Egypt culture values Honour, Loyalty, ...',
                'scraped_at': '2025-12-06T00:42:39.201228'
            }
        """
        country_normalized = self._normalize_country(country)
        
        logger.info(f"Querying Cultural Atlas for: {country}")
        
        # Check if country exists
        if country_normalized not in self.data:
            logger.warning(f"Country not found in Cultural Atlas: {country}")
            
            # Track not found
            self.not_found_countries.add(country)
            
            result = {
                'retrieved': False,
                'country': country,
                'country_normalized': country_normalized,
                'url': None,
                'axes': {},
                'summary': None,
                'error': f'Country "{country}" not found in Cultural Atlas'
            }
            self._log_call(result)
            return result
        
        # Get country data
        country_data = self.data[country_normalized]
        
        # Extract axes (raw data from JSON)
        axes = country_data.get('axes', {})
        
        # Extract core values for minimal summary
        core_concepts = axes.get('Core Concepts', {}).get('content', {})
        core_values = core_concepts.get('core_values', [])
        
        # Build result - return RAW data for LLM to process
        result = {
            'retrieved': True,
            'country': country_data.get('country', country),
            'country_normalized': country_normalized,
            'url': country_data.get('url', ''),
            'axes': axes,  # Raw axes data - LLM will interpret
            'summary': self._minimal_summary(country_data.get('country', country), core_values),
            'scraped_at': country_data.get('scraped_at', '')
        }
        
        self._log_call(result)
        logger.debug(f"Successfully retrieved {len(axes)} axes for {country}")
        
        return result
    
    def _minimal_summary(self, country: str, core_values: List[str]) -> str:
        """
        Generate minimal summary - just list core values.
        
        LLM will do the real summarization when it reads the axes.
        
        Args:
            country: Country name
            core_values: List of core values
            
        Returns:
            Simple one-line summary
        """
        if core_values:
            values_str = ', '.join(core_values[:5])
            return f"{country} culture values {values_str}."
        return f"Cultural information available for {country}."
    
    def get_not_found_countries(self) -> List[str]:
        """
        Get list of countries that were queried but not found.
        
        Returns:
            Sorted list of country names
        """
        return sorted(list(self.not_found_countries))
    
    def get_available_countries(self) -> List[str]:
        """Get list of countries in database."""
        return sorted(self.data.keys())
    
    def is_country_available(self, country: str) -> bool:
        """
        Check if a country is in the database.
        
        Args:
            country: Country name to check
            
        Returns:
            True if country exists, False otherwise
        """
        country_normalized = self._normalize_country(country)
        return country_normalized in self.data


def test_cultural_atlas_tool():
    """Test the Cultural Atlas tool."""
    print("Testing Cultural Atlas Tool")
    print("=" * 70)
    
    tool = CulturalAtlasTool()
    
    # Test 1: Query existing country
    print("\nTest 1: Query Egypt")
    result = tool.call("egypt")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Country: {result.get('country')}")
    print(f"URL: {result.get('url', '')[:60]}...")
    print(f"Axes: {list(result.get('axes', {}).keys())}")
    print(f"Summary: {result.get('summary', '')}")
    
    # Check axes structure
    if result['retrieved']:
        axes = result['axes']
        if 'Core Concepts' in axes:
            core = axes['Core Concepts']
            print(f"\nCore Concepts structure:")
            print(f"  - Has 'content': {'content' in core}")
            print(f"  - Has 'url': {'url' in core}")
            if 'content' in core:
                content = core['content']
                print(f"  - Core values: {content.get('core_values', [])[:3]}")
    
    # Test 2: Country name variations
    print("\nTest 2: Country name variations")
    test_names = ["United_States", "USA", "south-korea", "UK"]
    for name in test_names:
        result = tool.call(name)
        print(f"  {name:20} → Retrieved: {result['retrieved']}, Country: {result.get('country', 'N/A')}")
    
    # Test 3: Non-existent country
    print("\nTest 3: Non-existent country")
    result = tool.call("Atlantis")
    print(f"Retrieved: {result['retrieved']}")
    print(f"Error: {result.get('error')}")
    print(f"Has 'available_countries_count': {'available_countries_count' in result}")
    
    # Test 4: Check not found list
    print("\nTest 4: Not found countries")
    not_found = tool.get_not_found_countries()
    print(f"Not found: {not_found}")
    
    # Test 5: Available countries
    print("\nTest 5: Available countries")
    available = tool.get_available_countries()
    print(f"Total available: {len(available)}")
    print(f"Sample: {available[:10]}")
    
    # Test 6: Verify field names match step2 expectations
    print("\nTest 6: Verify field names for step2 compatibility")
    result = tool.call("Egypt")
    required_fields = ['retrieved', 'country', 'url', 'axes', 'summary']
    print(f"Required fields present:")
    for field in required_fields:
        present = field in result
        print(f"  - {field}: {'OK' if present else 'MISSING'}")
    
    # Should NOT have these fields
    wrong_fields = ['success', 'source_url', 'sections', 'available_countries_count']
    print(f"\nFields that should NOT be present:")
    for field in wrong_fields:
        present = field in result
        print(f"  - {field}: {'WRONG!' if present else 'Correct'}")
    
    # Test 7: Usage stats
    print("\nTest 7: Usage statistics")
    stats = tool.get_usage_stats()
    print(f"Total calls: {stats['total_calls']}")
    
    # Test 8: Caching (create second instance)
    print("\nTest 8: Testing cache (creating second instance)")
    tool2 = CulturalAtlasTool()
    result = tool2.call("Japan")
    print(f"Second instance works: {result['retrieved']}")
    
    print("\n" + "=" * 70)
    print("Cultural Atlas Tool Test Complete")


if __name__ == "__main__":
    test_cultural_atlas_tool()