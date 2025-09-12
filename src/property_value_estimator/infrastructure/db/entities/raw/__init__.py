"""
Raw data entities package.

Contains entities that preserve the original CSV data structure,
including duplicates and data quality issues.
"""

from .base import RawBase
from .house_sales import RawHouseSales
from .zipcode_demographics import RawZipcodeDemographics
from .future_house_example import RawFutureHouseExample

__all__ = [
    "RawBase",
    "RawHouseSales", 
    "RawZipcodeDemographics",
    "RawFutureHouseExample",
]
