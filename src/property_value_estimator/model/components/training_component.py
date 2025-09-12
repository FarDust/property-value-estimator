"""
KFP component for model training using ModelTrainer
"""
from kfp import dsl
from kfp.dsl import Dataset, Model
from typing import NamedTuple


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "."
    ]
)
def train_model_component(
    train_dataset: Dataset,
    model_name: str = "property_estimator",
    target_column: str = "price",
    feature_columns: list = None # type: ignore
) -> NamedTuple('TrainingOutputs', [('trained_model', Model)]): # type: ignore
    """
    KFP component for training a model using ModelTrainer
    
    Args:
        train_dataset: Training dataset
        model_name: Name for the trained model
        target_column: Name of target column
        feature_columns: List of feature column names, all columns by default
        
    Returns:
        NamedTuple with trained_model as Model artifact
    """
    import pandas as pd
    import cloudpickle
    from property_value_estimator.model.model.training import ModelTrainer
    
    # Load training data
    train_data = pd.read_parquet(train_dataset.path)
    
    # Create ModelTrainer instance
    trainer = ModelTrainer(
        model_name=model_name,
        input_data=train_data,
        target_column=target_column,
        feature_columns=feature_columns
    )
    
    # Train the model
    trained_model = trainer.train()
    
    # Create model artifact
    feature_columns = trainer.get_feature_columns()
    model_artifact = Model(
        uri=dsl.get_uri(suffix='trained_model'),
        metadata={
            'model_name': model_name,
            'training_samples': len(train_data),
            'features': len(feature_columns),
            'target_column': target_column,
            'model_type': 'sklearn_pipeline'
        }
    )
    
    # Save model using cloudpickle
    with open(model_artifact.path, 'wb') as f:
        cloudpickle.dump(trained_model, f)
    
    # Create the output namedtuple dynamically
    TrainingOutputs = NamedTuple('TrainingOutputs', [('trained_model', Model)])
    return TrainingOutputs(trained_model=model_artifact)
