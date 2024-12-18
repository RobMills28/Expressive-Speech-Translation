"""
Language-specific analysis module initialization.
"""
from .french import FrenchAnalyzer
from .german import GermanAnalyzer
from .italian import ItalianAnalyzer
from .portuguese import PortugueseAnalyzer
from .spanish import SpanishAnalyzer

__all__ = [
    'FrenchAnalyzer',
    'GermanAnalyzer',
    'ItalianAnalyzer',
    'PortugueseAnalyzer',
    'SpanishAnalyzer'
]