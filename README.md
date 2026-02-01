# Phishing Detection Benchmarking Framework

## Overview

This repository contains a modular machine learning framework designed to evaluate and compare various algorithms for phishing URL detection. Unlike ad-hoc exploratory scripts, this system is engineered to ensure **statistical validity**, **reproducibility**, and **comparability** across different model architectures (Deep Learning vs. Classical ML).

The framework addresses common pitfalls in academic benchmarking, such as data leakage, non-deterministic execution, and lack of hardware resource profiling. It supports automated hyperparameter sweeps and provides a unified interface for models ranging from PyTorch-based CNNs to GPU-accelerated classical algorithms.

## Installation

This project adheres to modern Python packaging standards using `pyproject.toml`. It is recommended to run the framework within a virtual environment.

### 1. Clone the repository

git clone <repository_url>
cd <repository_directory>

### 2. Set up the environment

### On Linux / MacOS:

python -m venv venv
source venv/bin/activate

### On Windows:
python -m venv venv
.\venv\Scripts\Activate

### 3. Install dependencies

Install the package in editable mode. This ensures all dependencies (including torch, hydra-core, etc.) are resolved automatically based on the pyproject.toml configuration.

pip install -e .

Note: For GPU acceleration, ensure that the appropriate CUDA Toolkit version matching your PyTorch installation is available on your system.

### Usage

Standard Execution

To run a single benchmark using the default configuration defined in conf/config.yaml:

python run_benchmark.py


To run the full benchmark using the sweeper configuration defined in conf/config.yaml:

python run_benchmark.py -m

### Results & Analysis

Metric aggregation is handled automatically. Raw performance data is serialized to benchmark_results.json in the root directory.

To generate comparative reports and visualizations:

python analyze_results.py

This script will:

    Parse the accumulated JSON logs.

    Print a statistical summary to the console (grouping by model type).

    Generate plots in the outputs/ directory, including:

        Performance vs. Efficiency: Visualizing the trade-off between F1-Score and Inference Time.

        Resource Utilization: Comparing Peak VRAM and CPU usage across architectures.

        A/B Testing: Comparative charts between different dataset sources (e.g., Standard vs. PhiUSIIL) if available.

License

This project is intended for academic and research purposes.
