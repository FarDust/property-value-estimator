#!/bin/bash
set -e

echo "Starting MLflow model server..."

# Initialize pyenv
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:/root/.local/bin:$PATH"
eval "$(pyenv init -)"

# Start MLflow model serving (environment already prepared during build)
exec uv tool run mlflow models serve \
    --model-uri "file://${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 5000 \
    --no-conda
