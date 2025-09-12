"""
Raw house sales entity for initial data migration.

Preserves all original data including duplicate sales of the same property.
Uses composite key (id, date) to handle multiple sales of same property.
"""

from sqlalchemy import Integer, String, Float, DateTime, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from .base import RawBase


class RawHouseSales(RawBase):
    """Raw house sales data from kc_house_data.csv"""
    
    __tablename__ = "raw_house_sales"
    
    # Composite primary key to handle multiple sales of same property
    __table_args__ = (
        PrimaryKeyConstraint('id', 'date', name='pk_raw_house_sales'),
    )
    
    # Property identifier (can have multiple sales)
    id: Mapped[int] = mapped_column(Integer, nullable=False, comment="Property ID - can have multiple sales")
    
    # Sale information
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="Sale date")
    price: Mapped[float] = mapped_column(Float, nullable=False, comment="Sale price")
    
    # Property characteristics
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
        return f"<RawHouseSales(id={self.id}, date={self.date}, price={self.price})>"
