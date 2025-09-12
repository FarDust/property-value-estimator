"""
Kubeflow Pipeline for Property Value Estimator
"""

from kfp import dsl
from typing import List, Tuple, Any

from property_value_estimator.core.config import FEATURE_COLUMNS
from property_value_estimator.model.components.data_loader_component import data_loader_component
from property_value_estimator.model.components.feature_processor_component import feature_processor_component
from property_value_estimator.model.components.split_component import split_data_component
from property_value_estimator.model.components.training_component import train_model_component
from property_value_estimator.model.components.evaluation_component import evaluate_model_component
from property_value_estimator.model.components.serving_component import serving_component


# Pipeline metadata
PIPELINE_INFO = {
    "name": "Property Value Estimator Pipeline",
    "description": "End-to-end ML pipeline for property value estimation using house sales data",
    "version": "1.0.0"
}

# Component information
PIPELINE_COMPONENTS: List[Tuple[str, Any, str]] = [
    ("Data Loader", data_loader_component, "Load feature data from database using configurable SQL queries"),
    ("Data Split", split_data_component, "Split dataset into train/test/validation sets with stratification"),
    ("Model Training", train_model_component, "Train Random Forest regression model with hyperparameter tuning"),
    ("Model Evaluation", evaluate_model_component, "Evaluate model performance with metrics and visualizations"),
    ("Model Serving", serving_component, "Register trained model to MLflow for centralized serving")
]


def get_pipeline_info() -> dict:
    """Get pipeline information"""
    return PIPELINE_INFO.copy()


def get_component_info() -> List[Tuple[str, Any, str]]:
    """Get component information"""
    return PIPELINE_COMPONENTS.copy()


def create_pipeline_with_custom_image(base_image: str = "python:3.10-slim"):
    """
    Factory function to create a pipeline with custom base image for components
    
    Args:
        base_image: Docker image to use for all components
        
    Returns:
        Pipeline function with components using the specified base image
    """
    from property_value_estimator.model.components.data_loader_component import data_loader_component
    from property_value_estimator.model.components.split_component import split_data_component
    from property_value_estimator.model.components.training_component import train_model_component
    from property_value_estimator.model.components.evaluation_component import evaluate_model_component
    from property_value_estimator.model.components.serving_component import serving_component
    from property_value_estimator.model.components.feature_processor_component import feature_processor_component
    
    # Override base_image for all components if not using default
    if base_image != "python:3.10-slim":
        for component in [data_loader_component, split_data_component, train_model_component, 
                         evaluate_model_component, serving_component, feature_processor_component]:
            if hasattr(component, 'component_spec') and hasattr(component.component_spec, 'implementation'):
                component.component_spec.implementation.container.image = base_image
    
    @dsl.pipeline(
        name="property-value-estimator-pipeline",
        description="End-to-end ML pipeline for property value estimation"
    )
    def custom_pipeline(
        random_state: int = 42,
        test_split: float = 0.2,
        validation_split: float = 0.1,
        target_column: str = "price",
        model_name: str = "property_estimator",
        cv_folds: int = 5,
        chart_width: int = 800,
        chart_height: int = 600,
        chart_template: str = "plotly_white",
        tracking_uri: str = "file://./mlruns",
        artifact_path: str = "model",
        experiment_name: str = "property_value_estimator"
    ):
        """
        Property value estimator ML pipeline with custom base image
        """
        

        # Step 1: Load data from database
        data_task = data_loader_component()

        # Step 2: Process features and target
        feature_task = feature_processor_component(
            input_dataset=data_task.output,
            target_column=target_column,
            feature_columns=FEATURE_COLUMNS
        )

        # Step 3: Split data into train/test/validation sets
        split_task = split_data_component(
            input_dataset=feature_task.outputs["processed_dataset"],
            random_state=random_state,
            test_split=test_split,
            validation_split=validation_split,
            target_column=target_column
        )
        
        # Step 3: Train model
        train_task = train_model_component(
            train_dataset=split_task.outputs["train_data"],
            model_name=model_name,
            target_column=target_column,
            feature_columns=FEATURE_COLUMNS,
        )
        
        # Step 4: Evaluate model
        evaluate_task = evaluate_model_component(
            trained_model=train_task.outputs["trained_model"],
            test_dataset=split_task.outputs["test_data"],
            target_column=target_column,
            random_state=random_state,
            cv_folds=cv_folds,
            chart_width=chart_width,
            chart_height=chart_height,
            chart_template=chart_template
        )
        
        # Step 5: Register model to MLflow
        serving_task = serving_component(
            trained_model=train_task.outputs["trained_model"],
            train_dataset=split_task.outputs["train_data"],
            evaluation_metrics=evaluate_task.outputs["metrics"],
            tracking_uri=tracking_uri,
            artifact_path=artifact_path,
            experiment_name=experiment_name,
            model_name=model_name,
            target_column=target_column
        )
    
    return custom_pipeline


# Default pipeline using standard components
property_value_estimator_pipeline = create_pipeline_with_custom_image()
