# Docker

The image uses Python 3.14 and installs the project dependencies from `requirements.txt`. It contains no project source code. Mount the repository when starting a container.

Build the image from the repository root:

```bash
docker build -f docker/Dockerfile -t 1p1q .
```

Create local directories for saved models and evaluation results, then mount them with the JetClass dataset:

```bash
mkdir -p saved_models results

docker run --rm \
  -v "$PWD:/workspace" \
  -v /absolute/path/to/JetClass:/workspace/data/JetClass:ro \
  -v "$PWD/saved_models:/workspace/saved_models" \
  -v "$PWD/results:/workspace/results" \
  1p1q python train.py seed=run_001
```

Weights & Biases runs offline by default. For online logging, pass `WANDB_MODE=online` and `WANDB_API_KEY` with `docker run -e`.

Evaluate the saved run with the same mounts:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -v /absolute/path/to/JetClass:/workspace/data/JetClass:ro \
  -v "$PWD/saved_models:/workspace/saved_models" \
  -v "$PWD/results:/workspace/results" \
  1p1q python evaluate.py \
    --seed run_001 \
    --model-dir /workspace/saved_models
```

Configuration overrides can be appended to the training command in the usual `key=value` form.
