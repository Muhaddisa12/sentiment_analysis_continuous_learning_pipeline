"""
Database module for sentiment analysis system.
Handles database connections and data loading.
"""

from .db import get_connection

__all__ = ['get_connection']
