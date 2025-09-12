"""
Base classes for raw data entities.

Raw entities preserve the original data structure from CSV files,
including duplicates and any data quality issues.
"""

from sqlalchemy.orm import DeclarativeBase


class RawBase(DeclarativeBase):
    """Base class for all raw data entities."""
    pass
