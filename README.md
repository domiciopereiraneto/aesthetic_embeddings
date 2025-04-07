# Aesthetic Embeddings

This repository contains implementations of various algorithms for generating and optimizing embeddings for aesthetic image generation and evaluation. The algorithms leverage Stable Diffusion and aesthetic scoring models such as NIMA, LAION, and Simulacra Aesthetic Model (SAM).

## Cloning the Repository and Initializing Submodules

To clone the repository and initialize its submodules, follow these steps:

1. Clone the repository using the `--recurse-submodules` flag to ensure that all submodules are cloned along with the main repository:
    ```bash
    git clone --recurse-submodules <repository-url>
    ```

2. If you have already cloned the repository without the `--recurse-submodules` flag, you can initialize and update the submodules manually:
    ```bash
    git submodule update --init --recursive
    ```

3. To ensure that submodules are always updated to their latest committed state in the repository, you can run the following command whenever you pull new changes:
    ```bash
    git submodule update --recursive
    ```

By following these steps, you will have the repository and its submodules properly set up on your local machine.

## Conda Environment Setup

To set up the required environment, use the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate ldm
```

This will install all necessary dependencies, including PyTorch, Stable Diffusion, and other libraries required for the algorithms.

---

## Algorithms Overview

### 1. **Embedding Evolution with Adam Optimizer (`embedding_ev_adam.py`)**
This algorithm uses the Adam optimizer to iteratively refine text embeddings for generating aesthetically pleasing images. It supports gradient-based optimization and requires an aesthetic scoring model that supports backpropagation.

#### Execution:
```bash
python algorithms/embedding_ev_adam.py
```

---

### 2. **Embedding Evolution with CMA-ES (`embedding_ev_cmaes.py`)**
This algorithm uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize text embeddings. It is a gradient-free optimization method, making it suitable for scenarios where backpropagation is not supported.

#### Execution:
```bash
python algorithms/embedding_ev_cmaes.py
```

---

### 3. **Hybrid Optimization (`embedding_ev_hybryd.py`)**
This algorithm combines CMA-ES and Adam optimization. It first uses CMA-ES for global exploration and then refines the embeddings locally using the Adam optimizer.

#### Execution:
```bash
python algorithms/embedding_ev_hybryd.py
```

---

### 4. **Partial Embedding Evolution with CMA-ES (`embedding_ev_cmaes_partial.py`)**
This variant of the CMA-ES algorithm optimizes only a subset of the text embeddings while keeping the rest fixed. It is useful for fine-tuning specific parts of the embedding space.

#### Execution:
```bash
python algorithms/embedding_ev_cmaes_partial.py
```

---

## Configuration

Each algorithm uses a YAML configuration file located in the config directory. These files define parameters such as:

- Seed values
- Model ID
- Number of inference steps
- Guidance scale
- Image dimensions
- Optimization parameters (e.g., learning rate, population size, etc.)

Modify the respective configuration file (e.g., `config_adam.yaml`, `config_cmaes.yaml`) to customize the behavior of the algorithms.

---
