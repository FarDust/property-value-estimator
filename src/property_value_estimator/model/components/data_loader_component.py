"""
Kubeflow Pipeline component for data loading
"""

from kfp import dsl
from kfp.dsl import Dataset


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "."
    ]
)
def data_loader_component() -> Dataset:
    """
    Kubeflow Pipeline component for loading feature data from database.
    
    Returns:
        Dataset: Loaded feature data as a dataset artifact
    """
    from property_value_estimator.model.preprocessing import DataLoader
    
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    dataset_artifact = Dataset(
        uri=dsl.get_uri(suffix='feature_data.parquet'),
        metadata={
            'samples': len(df),
            'features': len(df.columns) - 1,
            'data_type': 'feature_data'
        }
    )
    
    df.to_parquet(dataset_artifact.path, index=False)
    
    return dataset_artifact
