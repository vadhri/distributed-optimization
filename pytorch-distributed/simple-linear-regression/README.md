# PyTorch Distributed Simple Linear Regression

This folder contains code and results for distributed linear regression using PyTorch's distributed computing capabilities. The implementation demonstrates how to train a linear regression model across multiple processes and devices, collect profiling and loss statistics, and save model checkpoints for each rank.

## Contents

| File/Folder | Description |
|-------------|------------|
| `distributed_linear_fit.py` | Main script for distributed linear regression training. |
| `distributed_linear_fit_with_profiling.py` | Training script with additional profiling and performance measurement. |
| `ring_reduce.py` | Implements ring-based reduction for distributed averaging. |
| `pytorch-distributed-across-cpu.ipynb` | Jupyter notebook demonstrating distributed training across CPUs. |
| `final_model.pt` | Final trained model checkpoint. |
| `model_rank_*.pt` | Model checkpoints for each rank/process. |
| `losses_rank_*.pt` | Loss values recorded for each rank during training. |
| `stats_rank_*.pt` | Training statistics for each rank. |
| `profiler_rank_*/` | Profiling results for each rank. |

## Usage

- Run `distributed_linear_fit.py` or `distributed_linear_fit_with_profiling.py` with PyTorch distributed launch utilities to start training across multiple processes.
- Use the notebook for interactive exploration and demonstration.
- Inspect the `*_rank_*.pt` files for per-rank results and statistics.

## Requirements
- Python 3.x
- PyTorch
- (Optional) Jupyter Notebook

## References
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)

---
For more details, see the scripts and notebook in this folder.
