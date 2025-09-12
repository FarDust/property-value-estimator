import yaml
"""
CLI for MLflow Model Server using Typer

This module provides a command-line interface for starting an MLflow model server
using the latest available registered model in MLflow.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import mlflow
import typer
from mlflow.tracking import MlflowClient

from property_value_estimator.core.settings import settings

app = typer.Typer(help="MLflow Model Server CLI for Property Value Estimator")


def get_latest_model_id() -> Optional[str]:
    """
    Get the latest available model ID from the mlruns directory.
    
    Returns:
        Latest model ID in format "m-xyz", or None if no model found
    """
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        if not models:
            typer.echo("❌ No registered models found", err=True)
            return None
        model = models[0]
        versions = client.get_latest_versions(model.name, stages=["Production", "Staging", "None"])
        if not versions:
            typer.echo("❌ No model versions found", err=True)
            return None
        latest_version = max(versions, key=lambda v: int(v.version))
        source_path = latest_version.source
        # Extract m-xxxxxx from source path
        import re
        match = re.search(r"m-[a-f0-9]+", source_path)
        if not match:
            typer.echo(f"❌ Could not extract model artifact ID from source path: {source_path}", err=True)
            return None
        model_id = match.group(0)
        typer.echo(f"🎯 Found latest model artifact ID: {model_id}")
        return model_id
    except Exception as e:
        typer.echo(f"Error finding latest model: {e}", err=True)
        return None


def get_available_models(client: MlflowClient) -> list[str]:
    """
    Get list of all available registered models.
    
    Args:
        client: MLflow client instance
        
    Returns:
        List of model names
    """
    try:
        registered_models = client.search_registered_models()
        return [str(model.name) for model in registered_models if model.name]  # type: ignore
    except Exception as e:
        typer.echo(f"Error listing registered models: {e}", err=True)
        return []


@app.command()
@app.command()
def serve(
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Name of the registered model to serve. If not provided, will use the configured default or first available model."
    ),
    host: str = typer.Option(
        settings.mlflow.serve_host,
        "--host",
        "-h",
        help="Host to bind the server to"
    ),
    port: int = typer.Option(
        settings.mlflow.serve_port,
        "--port",
        "-p",
        help="Port to bind the server to"
    ),
    workers: int = typer.Option(
        settings.mlflow.serve_workers,
        "--workers",
        "-w",
        help="Number of workers to use"
    ),
    timeout: int = typer.Option(
        settings.mlflow.serve_timeout,
        "--timeout",
        "-t",
        help="Timeout in seconds for requests"
    ),
    tracking_uri: Optional[str] = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI. If not provided, will use configured default."
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Patch meta.yaml artifact_location for local mode (replace /app/ with ./)"
    )
):
    """
    Start MLflow model server with the latest available registered model.
    """
    def patch_meta_yaml_for_local(model_id: str):
        """Patch meta.yaml artifact_location for local mode, with chown if needed."""
        import os
        meta_path = None
        for root, dirs, files in os.walk("mlruns"):
            for d in dirs:
                if d == model_id:
                    meta_path = os.path.join(root, d, "meta.yaml")
                    break
        if meta_path and os.path.exists(meta_path):
            # Check write permission
            if not os.access(meta_path, os.W_OK):
                typer.echo(f"🔒 Permission denied for {meta_path}. Requesting sudo to chown mlruns...")
                import getpass
                user = getpass.getuser()
                import subprocess
                try:
                    subprocess.run(["sudo", "chown", "-R", user, "mlruns"], check=True)
                    typer.echo("✅ mlruns ownership updated. Retrying patch...")
                except Exception as e:
                    typer.echo(f"❌ Failed to chown mlruns: {e}", err=True)
                    return
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
            if "artifact_location" in meta and meta["artifact_location"].startswith("file:///app/"):
                from pathlib import Path
                mlruns_path = Path(".").absolute().resolve()
                new_prefix = f"file://{mlruns_path}/"
                meta["artifact_location"] = meta["artifact_location"].replace("file:///app/", new_prefix)
                with open(meta_path, "w") as f:
                    yaml.safe_dump(meta, f)
                typer.echo(f"🔧 Patched artifact_location in {meta_path} to {meta['artifact_location']}")
            else:
                typer.echo(f"ℹ️ No patch needed for {meta_path}")
        else:
            typer.echo(f"❌ meta.yaml not found for model {model_id}", err=True)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        typer.echo(f"Using tracking URI: {tracking_uri}")
    else:
        configured_uri = settings.mlflow.tracking_uri
        if configured_uri.startswith("file://./"):
            mlruns_path = Path(configured_uri.replace("file://./", "./")).resolve()
            if not mlruns_path.exists():
                typer.echo(f"❌ MLruns path does not exist: {mlruns_path}", err=True)
                raise typer.Exit(1)
            absolute_uri = f"file://{mlruns_path}"
            mlflow.set_tracking_uri(absolute_uri)
            typer.echo(f"Using tracking URI: {absolute_uri}")
        else:
            mlflow.set_tracking_uri(configured_uri)
            typer.echo(f"Using tracking URI: {configured_uri}")
    
    # Create MLflow client
    try:
        client = MlflowClient()
    except Exception as e:
        typer.echo(f"❌ Failed to create MLflow client: {e}", err=True)
        raise typer.Exit(1)
    
    # Get available models
    available_models = get_available_models(client)
    if not available_models:
        typer.echo("❌ No registered models found in MLflow.", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📋 Found {len(available_models)} registered model(s): {', '.join(available_models)}")
    
    if model_name is None:
        configured_model_name = settings.mlflow.model_name
        if configured_model_name in available_models:
            model_name = configured_model_name
            typer.echo(f"🎯 Using configured default model: {model_name}")
        else:
            model_name = available_models[0]
            typer.echo(f"🎯 Using first available model: {model_name}")
    elif model_name not in available_models:
        typer.echo(f"❌ Model '{model_name}' not found.", err=True)
        typer.echo(f"Available models: {', '.join(available_models)}", err=True)
        raise typer.Exit(1)
    else:
        typer.echo(f"🎯 Using specified model: {model_name}")
    
    model_id = get_latest_model_id()
    if model_id is None:
        typer.echo("❌ No model artifacts found.", err=True)
        raise typer.Exit(1)
    if local:
        patch_meta_yaml_for_local(model_id)
    model_uri = f"models:/{model_id}"
    typer.echo("🚀 Starting MLflow model server...")
    typer.echo(f"   Model URI: {model_uri}")
    typer.echo(f"   Host: {host}")
    typer.echo(f"   Port: {port}")
    typer.echo(f"   Workers: {workers}")
    typer.echo(f"   Timeout: {timeout}s")
    # Build MLflow serve command
    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--no-conda"
    ]
    typer.echo(f"📝 Running: {' '.join(cmd)}")
    # Start server - fail fast
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        typer.echo("\n⏹️  Server stopped by user.")
        raise typer.Exit(0)
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ MLflow server failed with exit code {e.returncode}", err=True)
        raise typer.Exit(e.returncode)


@app.command()
def list_models():
    """
    List all available registered models in MLflow.
    """
    # Try to infer tracking URI from mlruns directory if it exists
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        absolute_path = mlruns_path.resolve()
        mlflow.set_tracking_uri(f"file://{absolute_path}")
        typer.echo(f"Using local tracking URI: file://{absolute_path}")
    
    client = MlflowClient()
    
    try:
        registered_models = client.search_registered_models()
        
        if not registered_models:
            typer.echo("📭 No registered models found.")
            return
        
        typer.echo(f"📋 Found {len(registered_models)} registered model(s):")
        typer.echo()
        
        for model in registered_models:
            typer.echo(f"🔹 {model.name}")  # type: ignore
            
            # Get model versions
            try:
                versions = client.get_latest_versions(str(model.name), stages=["None", "Staging", "Production"])  # type: ignore
                if versions:
                    latest_version = max(versions, key=lambda x: int(x.version))
                    typer.echo(f"   Latest version: {latest_version.version}")
                    typer.echo(f"   Stage: {latest_version.current_stage}")
                else:
                    typer.echo("   No versions found")
            except Exception as e:
                typer.echo(f"   Error getting versions: {e}")
            
            typer.echo()
            
    except Exception as e:
        typer.echo(f"❌ Error listing models: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def model_info(
    model_name: str = typer.Argument(..., help="Name of the registered model"),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Model version. If not provided, will show info for the latest version."
    )
):
    """
    Show detailed information about a specific model.
    """
    # Try to infer tracking URI from mlruns directory if it exists
    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        absolute_path = mlruns_path.resolve()
        mlflow.set_tracking_uri(f"file://{absolute_path}")
    
    client = MlflowClient()
    
    try:
        # Get model version
        if version is None:
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            if not latest_versions:
                typer.echo(f"❌ No versions found for model '{model_name}'.", err=True)
                raise typer.Exit(1)
            model_version = max(latest_versions, key=lambda x: int(x.version))
            version = model_version.version
        else:
            model_version = client.get_model_version(model_name, version)
        
        typer.echo(f"📊 Model Information: {model_name} (version {version})")
        typer.echo("=" * 50)
        typer.echo(f"Stage: {model_version.current_stage}")
        typer.echo(f"Status: {model_version.status}")
        typer.echo(f"Run ID: {model_version.run_id}")
        typer.echo(f"Source: {model_version.source}")
        typer.echo(f"Creation Time: {model_version.creation_timestamp}")
        
        if model_version.description:
            typer.echo(f"Description: {model_version.description}")
        
        # Get run information
        try:
            if model_version.run_id:
                run = client.get_run(model_version.run_id)
                typer.echo("\n📈 Run Metrics:")
                for key, value in run.data.metrics.items():  # type: ignore
                    typer.echo(f"  {key}: {value}")
                
                typer.echo("\n⚙️  Run Parameters:")
                for key, value in run.data.params.items():  # type: ignore
                    typer.echo(f"  {key}: {value}")
                
        except Exception as e:
            typer.echo(f"\n⚠️  Could not retrieve run information: {e}")
            
    except Exception as e:
        typer.echo(f"❌ Error getting model info: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
