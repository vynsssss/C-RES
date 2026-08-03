# -*- coding: utf-8 -*-
"""
Hofstede Cultural Dimensions Tool - OPTIMIZED VERSION

Retrieves Hofstede cultural dimension scores for countries.
Dimensions: Power Distance, Individualism, Masculinity, Uncertainty Avoidance,
Long-term Orientation, Indulgence

"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from base_tool import BaseTool

logger = logging.getLogger(__name__)


class HofstedeTool(BaseTool):
    """
    Tool for querying Hofstede cultural dimensions.
    
    """
    
    # Class-level cache (shared across all instances)
    _data_cache = None
    _cache_path = None
    
    def __init__(self, data_path: Optional[str] = None):
        super().__init__()
        
        # Determine data path
        if data_path is None:
            data_path = Path(__file__).parent / "hofstede_dimensions.csv"
        else:
            data_path = Path(data_path)
        
        # Validate file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Hofstede data file not found: {data_path}")
        
        # Load from cache or read from file
        if (HofstedeTool._data_cache is None or 
            HofstedeTool._cache_path != str(data_path)):
            logger.info(f"Loading Hofstede data from {data_path}")
            self.data = pd.read_csv(data_path)
            self.data['country'] = self.data['country'].str.lower().str.strip()
            self.data.set_index('country', inplace=True)
            self.data.drop(columns=['ctr'], inplace=True)
            
            # Cache for future instances
            HofstedeTool._data_cache = self.data.copy()
            HofstedeTool._cache_path = str(data_path)
            logger.info(f"Cached Hofstede data ({len(self.data)} countries)")
        else:
            logger.debug("Using cached Hofstede data")
            self.data = HofstedeTool._data_cache.copy()
        
        # Dimension explanations
        self.dimension_info = {
            'power_distance': 'Extent to which less powerful members accept unequal power distribution',
            'individualism': 'Degree of interdependence among society members (vs collectivism)',
            'masculinity': 'Preference for achievement, assertiveness (vs cooperation, caring)',
            'uncertainty_avoidance': 'Extent to which members feel threatened by ambiguous situations',
            'long_term_orientation': 'Focus on future rewards (vs short-term tradition/norms)',
            'indulgence': 'Extent to which people try to control desires and impulses'
        }
        
        # Score interpretation functions (dispatch dictionary)
        self.interpreters = {
            'power_distance': self._interpret_power_distance,
            'individualism': self._interpret_individualism,
            'masculinity': self._interpret_masculinity,
            'uncertainty_avoidance': self._interpret_uncertainty_avoidance,
            'long_term_orientation': self._interpret_long_term_orientation,
            'indulgence': self._interpret_indulgence,
        }
    
    @property
    def name(self) -> str:
        return "hofstede_tool"
    
    @property
    def description(self) -> str:
        return (
            "Query Hofstede cultural dimensions for a country. "
            "Returns scores (0-100) for power distance, individualism, masculinity, "
            "uncertainty avoidance, long-term orientation, and indulgence."
        )
    
    # Country name mapping for common variants
    COUNTRY_ALIASES = {
        'united states': 'u.s.a.',
        'usa': 'u.s.a.',
        'us': 'u.s.a.',
        'united states of america': 'u.s.a.',
        'america': 'u.s.a.',
        'turkey': 'turkey',
        'turkiye': 'turkey',
        't\u00fcrkiye': 'turkey',
        'south korea': 'south korea',
        'korea': 'south korea',
        'czech republic': 'czech rep',
        'slovak republic': 'slovak rep',
        'dominican republic': 'dominican rep',
        'united kingdom': 'united kingdom',
        'uk': 'united kingdom',
        'great britain': 'united kingdom',
        'britain': 'united kingdom',
        'england': 'united kingdom',
        'hong kong': 'hong kong',
        'south africa': 'south africa',
        'new zealand': 'new zealand',
        'puerto rico': 'puerto rico',
        'saudi arabia': 'saudi arabia',
        'trinidad and tobago': 'trinidad and tobago',
        'burkina faso': 'burkina faso',
        'bosnia and herzegovina': 'bosnia',
    }
    
    def _normalize_country(self, country: str) -> str:
        """
        Normalize country name for lookup.
        Handles: hyphens, underscores, case, extra spaces, and common name variants
        
        Args:
            country: Country name
            
        Returns:
            Normalized country name
        """
        normalized = country.lower().strip()
        normalized = normalized.replace('_', ' ')
        normalized = normalized.replace('-', ' ')  # Handle hyphens
        
        # Check for known aliases
        if normalized in self.COUNTRY_ALIASES:
            normalized = self.COUNTRY_ALIASES[normalized]
        
        return normalized
    
    def call(self, country: str, dimensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get Hofstede scores for a country.
        
        Args:
            country: Country name (case-insensitive)
            dimensions: Optional list of specific dimensions to retrieve.
                       If None, returns all dimensions.
        
        Returns:
            Dict with cultural dimension scores and explanations
        """
        country_normalized = self._normalize_country(country)
        
        logger.info(f"Querying Hofstede data for: {country}")
        
        # Check if country exists
        if country_normalized not in self.data.index:
            logger.warning(f"Country not found in Hofstede database: {country}")
            result = {
                'success': False,
                'country': country,
                'error': f'Country "{country}" not found in Hofstede database',
                #'available_countries': sorted(self.data.index.tolist()),
                'suggestion': self._suggest_similar_country(country_normalized)
            }
            self._log_call(result)
            return result
        
        # Get scores
        scores = self.data.loc[country_normalized].to_dict()
        
        # Filter dimensions if specified
        if dimensions:
            # Validate requested dimensions
            invalid_dims = [d for d in dimensions if d not in scores]
            if invalid_dims:
                logger.warning(f"Invalid dimensions requested: {invalid_dims}")
                result = {
                    'success': False,
                    'country': country,
                    'error': f'Invalid dimensions: {invalid_dims}',
                    'valid_dimensions': list(scores.keys())
                }
                self._log_call(result)
                return result
            
            scores = {k: v for k, v in scores.items() if k in dimensions}
        
        # Build result
        result = {
            'success': True,
            'country': country,
            'country_normalized': country_normalized,
            'scores': scores,
            # NOTE: 'interpretations' is computed here for convenience/logging but is
            # NOT shown to the model. The coordinator's _format_hofstede() renders
            # ONLY the raw numeric scores ("Individualism: 75/100"), matching the
            # paper (Appendix B). The interpretation strings never enter the prompt.
            'interpretations': self._interpret_scores(scores),
            'dimension_info': {k: self.dimension_info[k] for k in scores.keys() if k in self.dimension_info}
        }
        
        self._log_call(result)
        logger.debug(f"Successfully retrieved {len(scores)} dimensions for {country}")
        
        return result
    
    def _suggest_similar_country(self, country: str) -> Optional[str]:
        """
        Suggest a similar country name if exact match not found.
        
        Args:
            country: Country name that wasn't found
            
        Returns:
            Suggested country name or None
        """
        available = self.data.index.tolist()
        
        # Simple substring matching
        matches = [c for c in available if country in c or c in country]
        
        if matches:
            return matches[0]
        
        return None
    
    def _interpret_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """
        Interpret Hofstede scores into readable text.
        
        Args:
            scores: Dict of dimension scores
            
        Returns:
            Dict of interpretations
        """
        return {
            dim: self.interpreters.get(dim, lambda s: f"Score: {s}")(score)
            for dim, score in scores.items()
        }
    
    def _interpret_power_distance(self, score: float) -> str:
        """Interpret power distance score."""
        if score >= 70:
            return "High power distance - hierarchical, accept inequality"
        elif score >= 40:
            return "Medium power distance - some hierarchy accepted"
        else:
            return "Low power distance - equality valued, flat hierarchies"
    
    def _interpret_individualism(self, score: float) -> str:
        """Interpret individualism score."""
        if score >= 70:
            return "Individualistic - value personal freedom and achievement"
        elif score >= 40:
            return "Balanced individualism/collectivism"
        else:
            return "Collectivistic - value group harmony and loyalty"
    
    def _interpret_masculinity(self, score: float) -> str:
        """Interpret masculinity score."""
        if score >= 70:
            return "Masculine - competitive, achievement-oriented"
        elif score >= 40:
            return "Balanced masculine/feminine values"
        else:
            return "Feminine - cooperative, caring for others"
    
    def _interpret_uncertainty_avoidance(self, score: float) -> str:
        """Interpret uncertainty avoidance score."""
        if score >= 70:
            return "High uncertainty avoidance - prefer structure and rules"
        elif score >= 40:
            return "Medium uncertainty avoidance"
        else:
            return "Low uncertainty avoidance - comfortable with ambiguity"
    
    def _interpret_long_term_orientation(self, score: float) -> str:
        """Interpret long-term orientation score."""
        if score >= 70:
            return "Long-term oriented - pragmatic, future-focused"
        elif score >= 40:
            return "Balanced time orientation"
        else:
            return "Short-term oriented - traditional, normative"
    
    def _interpret_indulgence(self, score: float) -> str:
        """Interpret indulgence score."""
        if score >= 70:
            return "Indulgent - allow gratification of desires"
        elif score >= 40:
            return "Balanced indulgence/restraint"
        else:
            return "Restrained - control gratification, strict social norms"
    
    def compare_countries(self, country1: str, country2: str) -> Dict[str, Any]:
        """
        Compare cultural dimensions between two countries.
        
        Args:
            country1: First country
            country2: Second country
            
        Returns:
            Dict with comparison results
        """
        logger.info(f"Comparing countries: {country1} vs {country2}")
        
        result1 = self.call(country1)
        result2 = self.call(country2)
        
        if not result1['success'] or not result2['success']:
            return {
                'success': False,
                'error': 'One or both countries not found',
                'country1_result': result1,
                'country2_result': result2
            }
        
        # Calculate differences
        differences = {}
        for dim in result1['scores'].keys():
            diff = abs(result1['scores'][dim] - result2['scores'][dim])
            differences[dim] = {
                'country1_score': result1['scores'][dim],
                'country2_score': result2['scores'][dim],
                'difference': diff,
                'significant': diff >= 20  # Consider 20+ points significant
            }
        
        return {
            'success': True,
            'country1': country1,
            'country2': country2,
            'differences': differences
        }
    
    def get_available_countries(self) -> List[str]:
        """Get list of countries in database."""
        return sorted(self.data.index.tolist())
    
    def is_country_available(self, country: str) -> bool:
        """
        Check if a country is in the database.
        
        Args:
            country: Country name to check
            
        Returns:
            True if country exists, False otherwise
        """
        country_normalized = self._normalize_country(country)
        return country_normalized in self.data.index


def test_hofstede_tool():
    """Test the Hofstede tool."""
    print("Testing Hofstede Tool (Optimized Version)")
    print("=" * 60)
    
    tool = HofstedeTool()
    
    # Test 1: Query single country
    print("\nTest 1: Query Italy")
    result = tool.call("italy")
    print(f"Success: {result['success']}")
    print(f"Power Distance: {result['scores']['power_distance']}")
    print(f"Interpretation: {result['interpretations']['power_distance']}")
    
    # Test 2: Query specific dimensions
    print("\nTest 2: Query USA (specific dimensions)")
    result = tool.call("united states", dimensions=['individualism', 'power_distance'])
    if result['success']:
        print(f"Scores: {result['scores']}")
    
    # Test 3: Country not found
    print("\nTest 3: Query non-existent country")
    result = tool.call("atlantis")
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    if result.get('suggestion'):
        print(f"Suggestion: {result['suggestion']}")
    
    # Test 4: Compare countries
    print("\nTest 4: Compare USA vs Japan")
    result = tool.compare_countries("united states", "japan")
    if result['success']:
        pd_diff = result['differences']['power_distance']
        print(f"Power Distance: {pd_diff['country1_score']} vs {pd_diff['country2_score']} (diff: {pd_diff['difference']})")
        print(f"Significant: {pd_diff['significant']}")
    
    # Test 5: Check availability
    print("\nTest 5: Check country availability")
    print(f"Italy available: {tool.is_country_available('italy')}")
    print(f"Atlantis available: {tool.is_country_available('atlantis')}")
    
    # Test 6: Usage stats
    print("\nTest 6: Usage statistics")
    stats = tool.get_usage_stats()
    print(f"Total calls: {stats['total_calls']}")
    
    # Test 7: Caching (create second instance)
    print("\nTest 7: Testing cache (creating second instance)")
    tool2 = HofstedeTool()
    result = tool2.call("germany")
    print(f"Second instance works: {result['success']}")
    
    print("\n" + "=" * 60)
    print("Hofstede Tool Test Complete")


if __name__ == "__main__":
    test_hofstede_tool()