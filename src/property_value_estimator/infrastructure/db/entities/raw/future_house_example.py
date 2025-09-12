"""
Raw future house examples entity for initial data migration.

These are unlabeled property examples for prediction.
Uses auto-generated ID since original data has no ID column.
"""

from sqlalchemy import Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column

from .base import RawBase


class RawFutureHouseExample(RawBase):
    """Raw future house examples data from future_unseen_examples.csv"""
    
    __tablename__ = "raw_future_house_examples"
    
    # Auto-generated primary key since original data has no ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="Auto-generated ID")
    
    # Property characteristics (same as house sales but no price/date)
    bedrooms: Mapped[int] = mapped_column(Integer, nullable=False, comment="Number of bedrooms")
    bathrooms: Mapped[float] = mapped_column(Float, nullable=False, comment="Number of bathrooms")
    sqft_living: Mapped[int] = mapped_column(Integer, nullable=False, comment="Square feet of living space")
    sqft_lot: Mapped[int] = mapped_column(Integer, nullable=False, comment="Square feet of lot")
    floors: Mapped[float] = mapped_column(Float, nullable=False, comment="Number of floors")
    waterfront: Mapped[int] = mapped_column(Integer, nullable=False, comment="Waterfront property (0/1)")
    view: Mapped[int] = mapped_column(Integer, nullable=False, comment="View quality (0-4)")
    condition: Mapped[int] = mapped_column(Integer, nullable=False, comment="Property condition (1-5)")
    grade: Mapped[int] = mapped_column(Integer, nullable=False, comment="Building grade (1-13)")
    sqft_above: Mapped[int] = mapped_column(Integer, nullable=False, comment="Square feet above ground")
    sqft_basement: Mapped[int] = mapped_column(Integer, nullable=False, comment="Square feet of basement")
    yr_built: Mapped[int] = mapped_column(Integer, nullable=False, comment="Year built")
    yr_renovated: Mapped[int] = mapped_column(Integer, nullable=False, comment="Year renovated (0 if never)")
    
    # Location information
    zipcode: Mapped[str] = mapped_column(String(10), nullable=False, comment="ZIP code")
    lat: Mapped[float] = mapped_column(Float, nullable=False, comment="Latitude")
    long: Mapped[float] = mapped_column(Float, nullable=False, comment="Longitude")
    
    # Neighborhood characteristics
    sqft_living15: Mapped[int] = mapped_column(Integer, nullable=False, comment="Avg sqft living of 15 nearest neighbors")
    sqft_lot15: Mapped[int] = mapped_column(Integer, nullable=False, comment="Avg sqft lot of 15 nearest neighbors")
    
    def __repr__(self) -> str:
        return f"<RawFutureHouseExample(id={self.id}, bedrooms={self.bedrooms}, zipcode={self.zipcode})>"
