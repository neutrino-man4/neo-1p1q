# 1P1Q: Particle Physics Data Encoding for Jet Physics
<img src="https://etpwww.etp.kit.edu/~abal/icons/1p1q_logo.png" alt="alt text" width="15%">

**Authors:** Aritra Bal¹, Benedikt Maier², Melik Oughton², Eric Pezone²  
¹ Karlsruhe Institute of Technology (KIT), Germany  
² Imperial College London, UK

[![arXiv](https://img.shields.io/badge/arXiv-2502.17301-b31b1b.svg)](https://arxiv.org/abs/2502.17301)



## Quick Start

### Environment Setup

We provide a convenient setup script to create the conda environment and install all dependencies:

1. **Clone the repository and navigate to it**
2. **Run the setup script:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```
3. **Activate the environment:**
   ```bash
   conda activate quantum-cpu
   ```
4. **Set up environment variables:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

The setup script creates a conda environment named `quantum-cpu` with Python 3.10 and installs all required packages from `requirements.txt`.

### Data Access

Download the training data from [Google Drive](https://drive.google.com/drive/folders/1fGATNxxcCKPk6mZ54Ucv1mYZteOnh33-?usp=sharing).



## Training

### Multi-Core CPU Training

This repository uses [Hydra](https://hydra.cc/) for configuration management and [Weights & Biases](https://wandb.ai/) for experiment tracking.

**Basic training command:**
```bash
python3 train.py --config-path $HYDRA_CONF --config-name config
```

**Multi-core training setup:**
```bash
export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=<N_THREADS>
python3 train.py --config-path $HYDRA_CONF --config-name config
```

Set the `lightning.kokkos` device in your YAML configuration for multi-core execution. Example configuration files are available in the `hydra_configs/` directory.

Create a WandB account if you don't already have one, and set your API key using:

```bash
export WANDB_API_KEY=YOUR_KEY_HERE
```


### GPU Training (Recommended)

For accelerated training, GPU support is available with the following requirements:
- GPU with Compute Capability ≥ 7.0
- CUDA Version ≥ 12.0

**Docker Setup:**
```bash
docker pull neutrinoman4/qml-lightning.gpu:v5.0
```
> **Note:** Ensure you use version v5.0

**Alternative: Singularity/Apptainer:**
```bash
apptainer build qml-lightning-gpu.sif docker://neutrinoman4/qml-lightning.gpu:v5.0
apptainer shell qml-lightning-gpu.sif
```

**GPU training command:**
```bash
python3 train.py --config-path $HYDRA_CONF --config-name config device=lightning.gpu
```

### HPC Cluster Support

- **Horeka@KIT (Slurm):** See [documentation](https://www.nhr.kit.edu/userdocs/ftp/containers/)
- **HTCondor:** Example scripts available in `condor_example/`



## Testing and Inference

### CPU Testing (Recommended)

For optimal performance during testing, use single-threaded CPU execution:

```bash
OMP_NUM_THREADS=1 OMP_PROC_BIND=spread python3 test_jetclass.py --config-name VQC_JC_SAMPLE_10Q read_n=1000
```

Inference can run on both CPU and GPU, though GPU provides minimal performance benefits.


### Output Configuration

- **`read_n`**: Controls the number of samples used for testing
- **`plot_dir`**: Directory where all plots are saved
- **`dump_dir`**: Directory where output HDF5 files are saved
- **`log_wandb`**: Whether or not to log all images to the same WandB run used for training

Configure these parameters in your Hydra config file. Latest configuration examples are available in `hydra_configs/`.

---

## Project Structure

### Core Components

- **`train.py`**: Main training script
- **`test.py`** / **`test_jetclass.py`**: Inference and testing scripts
- **`case_reader.py`**: Data loader implementation
- **`quantum/architectures.py`**: Quantum circuit architectures
- **`quantum/losses.py`**: Loss function definitions
- **`hydra_configs/`**: Configuration examples

### Available Architectures

Currently implemented quantum circuit architectures:
- `circuit()`
- `QCNN_circuit()`

Feel free to implement and experiment with new architectures.

### Experiment Management

- **Seed Management**: The `seed` parameter identifies training runs, with results saved to `/path/to/base/directory/{seed}/`
- **Descriptions**: Use the `desc` argument to add detailed descriptions to runs
- **Architecture Freezing**: Training automatically saves a copy of `architectures.py` as `FROZEN_ARCHITECTURE.py` in the save directory



## Configuration

Example Hydra configuration structure:

```yaml
# Device configuration
device: lightning.kokkos  # or lightning.gpu

# Training parameters
seed: experiment_001
desc: "Description of experiment"

# Output directories
plot_dir: "./plots"
dump_dir: "./outputs"

# Testing parameters
read_n: 1000
log_wandb: true
```


There are many more parameters that can be controlled, take a look at the latest and greatest example configs in `hydra_configs/`



## Requirements

- Python 3.10
- PennyLane Lightning GPU v0.38.0 (for GPU support)
- Additional dependencies listed in `requirements.txt`



## Citation

If you use this code in your research, please cite:

```bibtex
@article{bal2025anomaly,
  title={Anomaly detection in high-energy physics using a quantum autoencoder},
  author={Bal, Aritra and Maier, Benedikt and Oughton, Melik and Pezone, Eric},
  journal={arXiv preprint arXiv:2502.17301},
  year={2025}
}
```