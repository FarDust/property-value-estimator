"""
KFP component for model serving using MLflowRegistry
"""
from kfp import dsl
from kfp.dsl import Model, Dataset, Metrics
from typing import Optional, Dict, NamedTuple


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "."
    ]
)
def serving_component(
    trained_model: Model,
    train_dataset: Dataset,
    evaluation_metrics: Metrics,
    tracking_uri: str,
    artifact_path: str = "model",
    experiment_name: str = "property_value_estimator",
    registry_uri: Optional[str] = None,
    model_name: str = "property_estimator",
    target_column: str = "price",
    tags: Optional[Dict[str, str]] = None
) -> NamedTuple('ServingOutputs', [('model_uri', str)]): # type: ignore
    """
    KFP component for registering a trained model to MLflow
    
    Args:
        trained_model: Trained model artifact
        train_dataset: Training dataset for schema inference
        evaluation_metrics: Evaluation metrics from model evaluation
        tracking_uri: MLflow tracking URI (file://, s3://, gcs://, etc.)
        artifact_path: Path to store model artifacts
        experiment_name: MLflow experiment name
        registry_uri: Model registry URI, defaults to tracking URI
        model_name: Name for the registered model
        target_column: Name of the target column
        tags: Optional tags for the model
        
    Returns:
        NamedTuple with model_uri as string
    """
    import cloudpickle
    import pandas as pd
    from property_value_estimator.model.serving.mlflow import MLflowRegistry
    
    # Load the trained model
    with open(trained_model.path, 'rb') as f:
        model = cloudpickle.load(f)
    
    # Load training data for schema inference
    train_data = pd.read_parquet(train_dataset.path)
    
    # Setup MLflow registry
    registry = MLflowRegistry(
        tracking_uri=tracking_uri,
        artifact_path=artifact_path,
        experiment_name=experiment_name,
        registry_uri=registry_uri
    )
    registry.setup()
    
    # Extract metadata for parameters
    metadata = trained_model.metadata or {}
    parameters = {
        "model_name": model_name,
        "training_samples": metadata.get("training_samples", len(train_data)),
        "features": metadata.get("features", len(train_data.columns) - 1),  # -1 for target column
        "target_column": target_column,
        "model_type": metadata.get("model_type", "sklearn_pipeline"),
        "feature_columns": str(train_data.drop(columns=[target_column]).columns.tolist())
    }
    
    # Extract real metrics from evaluation
    metrics = evaluation_metrics.metadata or {}
    
    # Register the model with schema
    model_uri = registry.register_model(
        model=model,
        model_name=model_name,
        parameters=parameters,
        metrics=metrics,
        tags=tags,
        train_data=train_data,
        target_column=target_column
    )
    
    # Create the output namedtuple dynamically
    ServingOutputs = NamedTuple('ServingOutputs', [('model_uri', str)])
    return ServingOutputs(model_uri=model_uri)
