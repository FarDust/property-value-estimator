"""
Data loader for property value estimator model
"""

import pandas as pd
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from sqlalchemy import create_engine, text
from property_value_estimator.core.settings import settings


class DataLoader(BaseModel):
    """Data loader class for executing SQL queries against the database following project patterns."""
    
    query_path: Path = Field(
        default_factory=lambda: Path(".").absolute().resolve() / "assets" / "queries" / "features" / "property_value_estimator.sql",
        description="Path to the SQL query file"
    )
    
    _engine: Any = PrivateAttr(default=None)
    
    @field_validator('query_path')
    @classmethod
    def validate_query_path(cls, v: Path) -> Path:
        """Validate that the query file exists"""
        if not v.exists():
            raise ValueError(f"Query file does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Query path is not a file: {v}")
        return v
    
    @property
    def engine(self):
        """Lazy initialization of database engine"""
        if self._engine is None:
            self._engine = create_engine(settings.database.uri)
        return self._engine
    
    def load_data(self) -> pd.DataFrame:
        """
        Load feature data using the predefined SQL query.
        
        Returns:
            pd.DataFrame: DataFrame containing features and target for model training
        """
        # Read SQL query from file
        with open(self.query_path, 'r') as f:
            query = f.read()
        
        # Execute query and return DataFrame
        return pd.read_sql(text(query), self.engine)