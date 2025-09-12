"""
Raw zipcode demographics entity for initial data migration.

Preserves all original demographic data by zipcode.
Uses zipcode as primary key since each zipcode should have unique demographics.
"""

from sqlalchemy import String, Float
from sqlalchemy.orm import Mapped, mapped_column

from .base import RawBase


class RawZipcodeDemographics(RawBase):
    """Raw zipcode demographics data from zipcode_demographics.csv"""
    
    __tablename__ = "raw_zipcode_demographics"
    
    # Use zipcode as primary key
    zipcode: Mapped[str] = mapped_column(String(10), primary_key=True, comment="ZIP code")
    
    # Population data
    ppltn_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Total population quantity")
    urbn_ppltn_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Urban population quantity")
    sbrbn_ppltn_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Suburban population quantity")
    farm_ppltn_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Farm population quantity")
    non_farm_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Non-farm population quantity")
    
    # Income data
    medn_hshld_incm_amt: Mapped[float] = mapped_column(Float, nullable=True, comment="Median household income amount")
    medn_incm_per_prsn_amt: Mapped[float] = mapped_column(Float, nullable=True, comment="Median income per person amount")
    hous_val_amt: Mapped[float] = mapped_column(Float, nullable=True, comment="Housing value amount")
    
    # Education data (quantities)
    edctn_less_than_9_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Education less than 9 years quantity")
    edctn_9_12_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Education 9-12 years quantity")
    edctn_high_schl_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="High school education quantity")
    edctn_some_clg_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Some college education quantity")
    edctn_assoc_dgre_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Associate degree quantity")
    edctn_bchlr_dgre_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Bachelor degree quantity")
    edctn_prfsnl_qty: Mapped[float] = mapped_column(Float, nullable=True, comment="Professional degree quantity")
    
    # Percentage data
    per_urbn: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage urban")
    per_sbrbn: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage suburban")
    per_farm: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage farm")
    per_non_farm: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage non-farm")
    per_less_than_9: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage education less than 9 years")
    per_9_to_12: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage education 9-12 years")
    per_hsd: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage high school")
    per_some_clg: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage some college")
    per_assoc: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage associate degree")
    per_bchlr: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage bachelor degree")
    per_prfsnl: Mapped[float] = mapped_column(Float, nullable=True, comment="Percentage professional degree")
    
    def __repr__(self) -> str:
        return f"<RawZipcodeDemographics(zipcode={self.zipcode}, population={self.ppltn_qty})>"
