from typing import Any, Optional
from pydantic import BaseModel, Field
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature
from pathlib import Path
import pandas as pd


class MLflowRegistry(BaseModel):
    tracking_uri: str = Field(..., description="MLflow tracking URI (file://, s3://, gcs://, etc.)")
    artifact_path: str = Field(..., description="Path to store model artifacts")
    experiment_name: str = Field(..., description="MLflow experiment name")
    registry_uri: Optional[str] = Field(None, description="Model registry URI, defaults to tracking URI")
    
    class Config:
        arbitrary_types_allowed = True
    
    def setup(self) -> None:
        if self.tracking_uri.startswith("file://"):
            tracking_path = Path(self.tracking_uri.replace("file://", ""))
            # Convert to absolute path to avoid permission issues
            tracking_path = tracking_path.resolve()
            tracking_path.mkdir(parents=True, exist_ok=True)
            # Update tracking URI with absolute path
            self.tracking_uri = f"file://{tracking_path}"
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        mlflow.set_experiment(self.experiment_name)
    
    def register_model(
        self,
        model: Any,
        model_name: str,  
        parameters: dict[str, Any],
        metrics: dict[str, float],
        tags: Optional[dict[str, str]] = None,
        train_data: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None
    ) -> str:
        with mlflow.start_run():
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            
            if tags:
                mlflow.set_tags(tags)
            
            # Prepare input data and schema if training data is provided
            signature = None
            input_example = None
            
            if train_data is not None and target_column is not None:
                # Get feature columns (exclude target column)
                feature_columns = [col for col in train_data.columns if col != target_column]
                input_data = train_data[feature_columns]
                
                # Create a small sample for input example (5 rows)
                input_example = input_data.head(5)
                
                # Generate predictions for output schema
                sample_predictions = model.predict(input_example.to_numpy())
                
                # Infer signature from input and output
                signature = infer_signature(input_example, sample_predictions)
            
            # Build log_model kwargs conditionally
            log_model_kwargs = {
                "sk_model": model,
                "name": self.artifact_path,
                "registered_model_name": model_name
            }
            
            if signature is not None:
                log_model_kwargs["signature"] = signature
            
            if input_example is not None:
                log_model_kwargs["input_example"] = input_example
            
            model_info = mlflow_sklearn.log_model(**log_model_kwargs)
            
            return model_info.model_uri
