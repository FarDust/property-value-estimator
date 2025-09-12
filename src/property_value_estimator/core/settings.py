"""
Module for environment variable management
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    uri: str = "sqlite:///./property_estimator.db"


class ApiSettings(BaseModel):
    """API configuration settings."""
    name: str = "Property Value Estimator API"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class ModelServiceSettings(BaseModel):
    """External model service configuration."""
    url: str = "http://localhost:8080"
    timeout: int = 30


class MLflowSettings(BaseModel):
    """MLflow configuration settings."""
    tracking_uri: str = "file://./mlruns"
    model_name: str = "property_estimator"
    serve_host: str = "0.0.0.0"
    serve_port: int = 8080
    serve_workers: int = 1
    serve_timeout: int = 60


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    database: DatabaseSettings = DatabaseSettings()
    api: ApiSettings = ApiSettings()
    model_service: ModelServiceSettings = ModelServiceSettings()
    mlflow: MLflowSettings = MLflowSettings()
    
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()