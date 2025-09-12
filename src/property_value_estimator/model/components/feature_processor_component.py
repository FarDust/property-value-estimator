
"""
KFP component for feature selection using FeatureProcessor
"""
from kfp import dsl
from kfp.dsl import Dataset
from typing import NamedTuple, List

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "."
    ]
)
def feature_processor_component(
    input_dataset: Dataset,
    target_column: str = "price",
    feature_columns: List[str] = None # type: ignore
) -> NamedTuple('FeatureProcessorOutputs', [('processed_dataset', Dataset)]): # type: ignore
    """
    KFP component for selecting features and target using FeatureProcessor
    Args:
        input_dataset: Input dataset
        target_column: Name of target column
        feature_columns: List of feature column names, all columns by default
    Returns:
        NamedTuple with processed_dataset as Dataset artifact
    """
    import pandas as pd
    from property_value_estimator.model.preprocessing.feature_processor import FeatureProcessor

    # Load input data
    data = pd.read_parquet(input_dataset.path)

    # Process features and target
    processor = FeatureProcessor(
        data=data,
        target_column=target_column,
        feature_columns=feature_columns
    )
    processed_df = processor.get_feature_target_dataframe()

    # Create output artifact
    processed_artifact = Dataset(
        uri=dsl.get_uri(suffix='processed_dataset'),
        metadata={
            'samples': len(processed_df),
            'features': len(processor.get_feature_columns()),
            'target_column': target_column
        }
    )
    processed_df.to_parquet(processed_artifact.path)

    FeatureProcessorOutputs = NamedTuple('FeatureProcessorOutputs', [('processed_dataset', Dataset)])
    return FeatureProcessorOutputs(processed_dataset=processed_artifact)
