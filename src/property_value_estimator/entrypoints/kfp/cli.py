"""
CLI for Kubeflow Pipeline execution
"""

from pathlib import Path
from typing import Optional
import subprocess

import typer
from kfp import compiler, local, dsl

from property_value_estimator.model.pipeline import (
    property_value_estimator_pipeline,
    create_pipeline_with_custom_image,
    get_pipeline_info,
    get_component_info
)


app = typer.Typer(help="KFP Pipeline CLI for Property Value Estimator")


@app.command()
def compile_pipeline(
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for compiled pipeline (default: pipeline.yaml)"
    )
):
    """Compile the KFP pipeline to YAML"""
    if output_file is None:
        output_file = Path("pipeline.yaml")
    
    typer.echo("🔧 Compiling Property Value Estimator Pipeline...")
    
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=property_value_estimator_pipeline,
        package_path=str(output_file)
    )
    
    typer.echo(f"✅ Pipeline compiled successfully to {output_file}")

from pathlib import Path
from typing import Optional

import typer
from kfp import compiler, local

from property_value_estimator.model.pipeline import (
    property_value_estimator_pipeline,
    get_pipeline_info,
    get_component_info
)


@app.command()
def run_local(
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    test_split: float = typer.Option(0.2, help="Test split proportion"),
    validation_split: float = typer.Option(0.1, help="Validation split proportion"),
    target_column: str = typer.Option("price", help="Target column name"),
    model_name: str = typer.Option("property_estimator", help="Model name"),
    cv_folds: int = typer.Option(5, help="Cross-validation folds"),
    output_dir: Path = typer.Option(Path("./pipeline_outputs"), help="Output directory for artifacts"),
    raise_on_error: bool = typer.Option(True, help="Raise on component error")
):
    """Run the pipeline locally using KFP local runner"""
    
    typer.echo("🚀 Initializing KFP local runner...")
    
    # Initialize KFP local runner
    local.init(
        runner=local.SubprocessRunner(),
        raise_on_error=raise_on_error,
        pipeline_root=str(output_dir.absolute())
    )
    
    typer.echo(f"📁 Pipeline root: {output_dir.absolute()}")
    typer.echo("🔧 Running Property Value Estimator Pipeline locally...")

    property_value_estimator_pipeline(
        random_state=random_state,
        test_split=test_split,
        validation_split=validation_split,
        target_column=target_column,
        model_name=model_name,
        cv_folds=cv_folds
    )


@app.command()
def run_docker(
    random_state: int = typer.Option(42, help="Random state for reproducibility"),
    test_split: float = typer.Option(0.2, help="Test split proportion"),
    validation_split: float = typer.Option(0.1, help="Validation split proportion"),
    target_column: str = typer.Option("price", help="Target column name"),
    model_name: str = typer.Option("property_estimator", help="Model name"),
    cv_folds: int = typer.Option(5, help="Cross-validation folds"),
    output_dir: Path = typer.Option(Path("./pipeline_outputs"), help="Output directory for artifacts"),
    raise_on_error: bool = typer.Option(True, help="Raise on component error")
):
    """Run the pipeline using KFP DockerRunner with custom component image"""
    
    # 1. Build component image
    typer.echo("🔨 Building component image...")
    subprocess.run([
        "docker", "build", 
        "-f", "docker/Dockerfile.component", 
        "-t", "property-estimator-components:latest", 
        "."
    ], check=True)
    
    # 2. Setup mounts - just mount current directory into /app
    try:
        import docker
    except ImportError:
        typer.echo("❌ docker package required. Install with: pip install docker")
        raise typer.Exit(1)
    
    current_dir = Path.cwd()
    mounts = [
        docker.types.Mount(
            target="/app",
            source=str(current_dir),
            type="bind"
        )
    ]
    
    # 3. Create pipeline with custom component image
    typer.echo("🔧 Creating pipeline with custom component images...")
    custom_image = "property-estimator-components:latest"
    custom_pipeline = create_pipeline_with_custom_image(base_image=custom_image)
    
    # 4. Initialize KFP DockerRunner
    typer.echo("🐳 Initializing KFP DockerRunner...")
    local.init(
        runner=local.DockerRunner(mounts=mounts),
        raise_on_error=raise_on_error,
        pipeline_root=str(output_dir.absolute())
    )
    
    typer.echo(f"📁 Pipeline root: {output_dir.absolute()}")
    typer.echo("🔧 Running Property Value Estimator Pipeline with Docker...")

    # 5. Run pipeline with custom components
    custom_pipeline(
        random_state=random_state,
        test_split=test_split,
        validation_split=validation_split,
        target_column=target_column,
        model_name=model_name,
        cv_folds=cv_folds
    )


@app.command()
def show_info():
    """Show pipeline information"""
    
    # Get pipeline info from the pipeline module
    pipeline_info = get_pipeline_info()
    components_info = get_component_info()
    
    typer.echo("🏠 " + pipeline_info["name"])
    typer.echo("=" * (len(pipeline_info["name"]) + 3))
    typer.echo(f"Description: {pipeline_info['description']}")
    typer.echo(f"Version: {pipeline_info['version']}")
    typer.echo("")
    
    typer.echo("Pipeline Components:")
    for i, (name, component_func, description) in enumerate(components_info, 1):
        # Try to get docstring from component, fallback to description
        try:
            component_doc = component_func.__doc__.strip().split('\n')[0] if component_func.__doc__ else description
        except AttributeError:
            component_doc = description
        typer.echo(f"  {i}. {name} - {component_doc}")
    
    typer.echo("")
    typer.echo("Available Commands:")
    
    # Get command info from functions with @app.command decorators
    commands = [
        ("compile-pipeline", "Compile the KFP pipeline to YAML"),
        ("run-local", "Run the pipeline locally using KFP local runner"),
        ("show-info", "Show this pipeline information")
    ]
    
    for command_name, description in commands:
        typer.echo(f"  {command_name} - {description}")


if __name__ == "__main__":
    app()
