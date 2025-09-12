"""
KFP component for data splitting using ModelSplit
"""
from kfp import dsl
from kfp.dsl import Dataset
from typing import NamedTuple


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "."
    ]
)
def split_data_component(
    input_dataset: Dataset,
    random_state: int = 42,
    test_split: float = 0.2,
    validation_split: float = None, # type: ignore
    feature_columns: list = None, # type: ignore
    target_column: str = "price"
) -> NamedTuple('SplitOutputs', [('train_data', Dataset), ('test_data', Dataset), ('validation_data', Dataset)]): # type: ignore
    """
    KFP component for splitting data into train/test/validation sets
    
    Args:
        input_dataset: Input dataset to split
        random_state: Random state for reproducibility
        test_split: Proportion for test split
        validation_split: Proportion for validation split (optional)
        feature_columns: List of feature columns (optional)
        target_column: Name of target column
    
    Returns:
        NamedTuple with train_data, test_data, validation_data datasets
    """
    import pandas as pd
    from property_value_estimator.model.model.split import ModelSplit
    
    # Load input data
    input_data = pd.read_parquet(input_dataset.path)
    
    # Create ModelSplit instance
    splitter = ModelSplit(
        input_data=input_data,
        random_state=random_state,
        test_split=test_split,
        validation_split=validation_split,
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    # Process the split
    train_df, test_df, val_df = splitter.process()
    
    # Create output artifacts
    train_artifact = Dataset(
        uri=dsl.get_uri(suffix='train_data.parquet'),
        metadata={
            'samples': len(train_df),
            'features': len(train_df.columns) - 1,
            'split_type': 'train'
        }
    )
    train_df.to_parquet(train_artifact.path, index=False)

    test_artifact = Dataset(
        uri=dsl.get_uri(suffix='test_data.parquet'),
        metadata={
            'samples': len(test_df),
            'features': len(test_df.columns) - 1,
            'split_type': 'test'
        }
    )
    test_df.to_parquet(test_artifact.path, index=False)    # Handle validation data (might be None)
    if val_df is not None:
        val_artifact = Dataset(
            uri=dsl.get_uri(suffix='validation_data.parquet'),
            metadata={
                'samples': len(val_df),
                'features': len(val_df.columns) - 1,
                'split_type': 'validation'
            }
        )
        val_df.to_parquet(val_artifact.path, index=False)
    else:
        # Create empty artifact if no validation split
        val_artifact = Dataset(
            uri=dsl.get_uri(suffix='validation_data.parquet'),
            metadata={
                'samples': 0,
                'features': 0,
                'split_type': 'validation_empty'
            }
        )
        # Create empty Parquet
        pd.DataFrame().to_parquet(val_artifact.path, index=False)
    
    # Create the output namedtuple dynamically
    SplitOutputs = NamedTuple('SplitOutputs', [('train_data', Dataset), ('test_data', Dataset), ('validation_data', Dataset)])
    return SplitOutputs(train_data=train_artifact, test_data=test_artifact, validation_data=val_artifact)
