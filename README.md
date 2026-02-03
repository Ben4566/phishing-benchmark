# Phishing Detection Benchmark

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Hydra](https://img.shields.io/badge/Config-Hydra-orange)
![License](https://img.shields.io/badge/License-Academic-green)

## Overview

This repository hosts a modular, high-performance machine learning framework designed to evaluate and compare various architectures for **Phishing URL Detection**.

Originally initiated as a group project in the 5th semester, this codebase represents the **benchmarking engine**. It unifies diverse initial implementations (CNN, XGBoost, SVM, LR) into a cohesive, object-oriented framework focusing on **reproducibility**, **hardware efficiency**, and **architectural scalability**.

## Installation

It is recommended to run the framework within a virtual environment.

## 1. Clone the repository

```bash
git clone <https://github.com/Ben4566/phishing-benchmark>
cd <https://github.com/Ben4566/phishing-benchmark>
```

### 2. Set up the environment

On Linux / MacOS:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows

```bash
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install dependencies

Install the package in editable mode. This ensures all dependencies (including torch, hydra-core, etc.) are resolved automatically based on the `pyproject.toml` configuration.

```bash
pip install -e .
```

Note: For GPU acceleration, ensure that the appropriate CUDA Toolkit version matching your PyTorch installation is available on your system.

### Usage

Standard Execution

To run a single benchmark using the default configuration defined in conf/config.yaml:

```bash
python run_benchmark.py
```

To run the full benchmark using the sweeper configuration defined in conf/config.yaml:

```bash
python run_benchmark.py -m
```

### Results & Analysis

Raw performance data is serialized to `benchmark_results.json` in the root directory.

To generate comparative reports and visualizations:

```bash
python analyze_results.py
```

## Authorship

Benchmark: Benedikt Lang
Intial Notebook implemantion of LR: Moritz Umlaut
Intial Notebook implemantion of XGBoost: Patrick
Intial Notebook implemantion of SVM: Marek Polzer
Intial Notebook implemantion of CNN: Benedikt Lang

### License

This project is intended for academic and research purposes.
