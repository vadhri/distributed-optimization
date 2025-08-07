# Distributed Optimization

## Introduction
This repository contains a collection of Jupyter Notebooks and resources focused on distributed optimization, convex analysis, and consensus algorithms. The materials are organized by topic and cover both foundational concepts and advanced methods, including the Augmented Lagrangian, ADMM, dual ascent, graph-based consensus, and more. The files only contains python code that can be executed and theory part. 

## Table of Contents


## Organization


### core_methods/

Foundational optimization methods, theory, and mathematical background.

| File | Description |
|------|-------------|
| [AugmentedLagrangian](core_methods/AugmentedLagrangian.ipynb) | Augmented Lagrangian method for constrained optimization. |
| [DualAscent](core_methods/DualAscent.ipynb) | Dual ascent method for distributed problems. |
| [MethodOfMultipliers](core_methods/MethodOfMultipliers.ipynb) | Method of multipliers: theory and algorithms. |
| [NN_Optimization](core_methods/NN_Optimization.ipynb) | Optimization methods for neural network training. |
| [RestrictingOptimizationSurface](core_methods/RestrictingOptimizationSurface.ipynb) | Restricting optimization to feasible regions. |
| [Stabiilty](core_methods/Stabiilty.ipynb) | Stability analysis in optimization. |
| [constrained_optimization](core_methods/constrained_optimization.ipynb) | Solving constrained optimization problems. |
| [continous_fixed_time_dgd](core_methods/continous_fixed_time_dgd.ipynb) | Continuous fixed-time distributed gradient descent. |
| [convergence](core_methods/convergence.ipynb) | Convergence analysis of optimization routines. |
| [gradient_fields](core_methods/gradient_fields.ipynb) | Exploration of gradient fields. |
| [least_squares_2nd_order](core_methods/least_squares_2nd_order_optmization_sine_cos.ipynb) | Second-order least squares optimization (sine/cosine). |
| [normalized_gradient](core_methods/normalized_gradient_.ipynb) | Normalized gradients in optimization. |
| [optimization_with_constraints](core_methods/optimization_with_constraints.ipynb) | Practical optimization under constraints. |
| [saddle_point](core_methods/saddle_point.ipynb) | Saddle points in optimization landscapes. |
| [convex_functions](core_methods/math/convex_functions.ipynb) | Properties and relevance of convex functions. |

### applications/ADMM/

Applications of ADMM in distributed and large-scale optimization problems.

| File | Description |
|------|-------------|
| [ATM](applications/ADMM/ATM.ipynb) | ADMM for distributed optimization in ATM networks. |
| [Distributed_linear_regression](applications/ADMM/Distributed_linear_regression.ipynb) | Distributed linear regression with ADMM. |
| [LogisticRegression](applications/ADMM/LogisticRegression.ipynb) | Logistic regression via ADMM. |
| [consensus_cuda](applications/ADMM/consensus_cuda.ipynb) | GPU-accelerated consensus with ADMM. |
| [sharing_power_grid](applications/ADMM/sharing_power_grid.ipynb) | Power grid sharing optimization using ADMM. |

### graph_consensus/

Consensus algorithms and dynamics on graphs, including average, min, max, and stochastic consensus in various network topologies.

| File | Description |
|------|-------------|
| [SchocasticAdjacency](graph_consensus/SchocasticAdjacency.ipynb) | Stochastic adjacency matrices in consensus. |
| [connected_graph_consensus](graph_consensus/connected_graph_avg_min_max_consensus.ipynb) | Average, min, and max consensus in connected graphs. |
| [decentralized_gradient](graph_consensus/distributed_decentrarlized_gradient_descent.ipynb) | Distributed decentralized gradient descent. |
| [decentralized_gradient_2nd](graph_consensus/distributed_decentrarlized_gradient_descent_2nd_order.ipynb) | Second-order decentralized gradient descent. |
| [fxts_consensus](graph_consensus/fxts_consensus.ipynb) | Finite-time consensus algorithms. |
| [laplacian_dynamics](graph_consensus/laplacian_dynamics.ipynb) | Laplacian dynamics in consensus algorithms. |
| [edp](graph_consensus/uncapacitated_capacitated_edp.ipynb) | Edge-disjoint paths in graph optimization. |
| [drone_direction_consensus](graph_consensus/undirected_unweighted_drone_direction_consensus.ipynb) | Drone direction consensus in undirected, unweighted graphs. |
| [weighted_graph_consensus](graph_consensus/undirected_weighted_connected_graph_avg_min_max_consensus.ipynb) | Consensus in undirected, weighted connected graphs. |

For more details, see the individual notebooks. Contributions and suggestions are welcome!

### pytorch-distributed/

| Folder | Description |
|--------|-------------|
| [simple-linear-regression](pytorch-distributed/simple-linear-regression/) | Distributed linear regression using PyTorch's distributed computing. |
| [distributed-schocastic-gradient-descent](pytorch-distributed/distributed-schocastic-gradient-descent/) | Distributed stochastic gradient descent with profiling and results. |

### ray/

| Folder | Description |
|--------|-------------|
| [decentralized-sgd](ray/decentralized-sgd/) | Decentralized SGD experiments and scripts using Ray. |

### flower/

Federated learning experiments using Flower framework.

| Location | Description |
|----------|-------------|
| [simple-linear-regression](flower/simple-linear-regression/) | Federated simple linear regression with Flower. |
| [simple-minst-partitioned-iid](flower/simple-minst-partitioned-iid/) | Federated MNIST classification with partitioned IID data using Flower. |