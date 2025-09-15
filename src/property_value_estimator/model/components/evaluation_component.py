"""
KFP component for model evaluation using ModelEvaluator
"""

from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, HTML
from typing import NamedTuple


@dsl.component(base_image="python:3.10-slim", packages_to_install=["."])
def evaluate_model_component(
    trained_model: Model,
    test_dataset: Dataset,
    target_column: str = "price",
    random_state: int = 42,
    cv_folds: int = 5,
    chart_width: int = 800,
    chart_height: int = 600,
    chart_template: str = "plotly_white",
) -> NamedTuple(
    "EvaluationOutputs",
    [
        ("metrics", Metrics),
        ("feature_importance", Dataset),
        ("residuals_chart", HTML),
        ("predictions_chart", HTML),
        ("feature_importance_chart", HTML),
        ("learning_curves_chart", HTML),
    ],
):  # type: ignore
    """
    KFP component for evaluating a trained model using ModelEvaluator

    Args:
        trained_model: Trained model artifact
        test_dataset: Test dataset for evaluation
        target_column: Name of target column
        random_state: Random state for reproducibility
        cv_folds: Number of cross-validation folds
        chart_width: Width of generated charts
        chart_height: Height of generated charts
        chart_template: Plotly template for charts

    Returns:
        NamedTuple with metrics, feature_importance dataset, and individual chart HTML artifacts
    """
    import pandas as pd
    import cloudpickle
    from property_value_estimator.model.model.evaluation import (
        ModelEvaluator,
        ChartConfig,
    )

    # Load the trained model
    with open(trained_model.path, "rb") as f:
        model = cloudpickle.load(f)

    # Load test data
    test_data = pd.read_parquet(test_dataset.path)

    # Create chart configuration
    chart_config = ChartConfig(
        width=chart_width, height=chart_height, template=chart_template
    )

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model=model,
        testing_data=test_data,
        target_column=target_column,
        random_state=random_state,
        cv_folds=cv_folds,
        chart_config=chart_config,
    )

    # Get evaluation results
    metrics_df, feature_importance_df, figures_dict = evaluator.evaluate()

    # Create metrics artifact
    metrics_dict = {}
    for _, row in metrics_df.iterrows():
        metrics_dict[row["metric"]] = float(row["value"])

    metrics_artifact = Metrics(uri=dsl.get_uri(suffix="metrics"), metadata=metrics_dict)

    # Create feature importance dataset artifact
    feature_importance_artifact = Dataset(
        uri=dsl.get_uri(suffix="feature_importance.parquet"),
        metadata={
            "features_count": len(feature_importance_df),
            "data_type": "feature_importance",
        },
    )
    feature_importance_df.to_parquet(feature_importance_artifact.path, index=False)

    # Create individual HTML artifacts for each chart
    residuals_html = HTML(uri=dsl.get_uri(suffix="residuals_chart.html"))
    figures_dict["residuals"].write_html(residuals_html.path, include_plotlyjs="cdn")

    predictions_html = HTML(uri=dsl.get_uri(suffix="predictions_chart.html"))
    figures_dict["prediction_vs_actual"].write_html(
        predictions_html.path, include_plotlyjs="cdn"
    )

    feature_importance_html = HTML(
        uri=dsl.get_uri(suffix="feature_importance_chart.html")
    )
    figures_dict["feature_importance"].write_html(
        feature_importance_html.path, include_plotlyjs="cdn"
    )

    learning_curves_html = HTML(uri=dsl.get_uri(suffix="learning_curves_chart.html"))
    figures_dict["learning_curves"].write_html(
        learning_curves_html.path, include_plotlyjs="cdn"
    )

    # Create the output namedtuple dynamically
    EvaluationOutputs = NamedTuple(
        "EvaluationOutputs",
        [
            ("metrics", Metrics),
            ("feature_importance", Dataset),
            ("residuals_chart", HTML),
            ("predictions_chart", HTML),
            ("feature_importance_chart", HTML),
            ("learning_curves_chart", HTML),
        ],
    )

    # Return outputs
    return EvaluationOutputs(
        metrics=metrics_artifact,
        feature_importance=feature_importance_artifact,
        residuals_chart=residuals_html,
        predictions_chart=predictions_html,
        feature_importance_chart=feature_importance_html,
        learning_curves_chart=learning_curves_html,
    )
