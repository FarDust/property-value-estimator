# Property Value Estimator

A machine learning model that predicts residential property values using house characteristics and demographic data.

## Features

- Predicts property prices based on house features (bedrooms, bathrooms, square footage, etc.)
- Incorporates local demographic data for improved accuracy
- Built with scikit-learn and MLflow tracking
- Kubeflow pipeline for training and deployment
- FastAPI service for real-time predictions

## Architecture

The model uses a combination of:

- House characteristics (living space, lot size, number of rooms)
- Location-based demographic features (population, income, urban/suburban distribution)
- Engineered features for better prediction accuracy

## Quick Start

### Prerequisites

- Python 3.10+
- uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd phdata

# Install dependencies
uv sync
```

### Training the Model

```bash
# Run the complete training pipeline
uv run kfp-pipeline run-local
```

### Making Predictions

```bash
# Start the prediction service
uv run phdata serve

# The API will be available at http://localhost:8000
# Visit http://localhost:8000/docs for interactive API documentation
```

## Project Structure

- `src/property_value_estimator/` - Main application code
- `assets/queries/` - SQL feature engineering queries
- `pipeline_outputs/` - Training pipeline results and artifacts
- `alembic/` - Database migration scripts

## Development

The project uses:

- **MLflow** for experiment tracking and model versioning
- **Kubeflow Pipelines** for orchestrating training workflows
- **FastAPI** for serving predictions
- **SQLAlchemy** with Alembic for data management
- **uv** for dependency management

## Model Performance

Training results and model metrics are tracked in MLflow and stored in the `pipeline_outputs/` directory after each training run.
