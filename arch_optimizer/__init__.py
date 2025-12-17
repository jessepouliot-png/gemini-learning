"""
Arch Linux Optimization Module using Gemini AI

This package provides tools for analyzing and optimizing Arch Linux systems
using Google's Gemini AI.
"""

from .system_info import SystemInfoCollector
from .gemini_client import GeminiClient
from .analyzer import ArchOptimizer, load_config

__version__ = '1.0.0'
__all__ = [
    'SystemInfoCollector',
    'GeminiClient', 
    'ArchOptimizer',
    'load_config'
]
