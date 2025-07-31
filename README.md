# Distributed Optimization

## Introduction
This repository contains a collection of Jupyter Notebooks and resources focused on distributed optimization, convex analysis, and consensus algorithms. The materials are organized by topic and cover both foundational concepts and advanced methods, including the Augmented Lagrangian, ADMM, dual ascent, graph-based consensus, and more. The files only contains python code that can be executed and theory part. 

## Table of Contents


## Organization

### core_methods/

Foundational optimization methods, theory, and mathematical background.

| Filename | Description |
|----------|-------------|
| AugmentedLagrangian.ipynb | Introduction and implementation of the Augmented Lagrangian method for constrained optimization, with examples and visualizations. |
| DualAscent.ipynb | Step-by-step development and application of the dual ascent method to distributed problems. |
| MethodOfMultipliers.ipynb | In-depth exploration of the method of multipliers, including theory and practical algorithms. |
| NN_Optimization.ipynb | Covers gradient-based and advanced optimization methods for neural network training. |
| RestrictingOptimizationSurface.ipynb | Techniques and examples for limiting optimization to specific surfaces or regions. |
| Stabiilty.ipynb | Stability analysis of optimization algorithms and systems. |
| constrained_optimization.ipynb | Methods for formulating and solving constrained optimization problems, with worked examples. |
| convergence.ipynb | Theoretical and empirical analysis of convergence in optimization routines. |
| gradient_fields.ipynb | Visual and mathematical exploration of gradient fields in optimization landscapes. |
| least_squares_2nd_order_optmization_sine_cos.ipynb | Second-order optimization for least squares problems involving sine and cosine functions. |
| normalized_gradient_.ipynb | Effects and applications of using normalized gradients in optimization. |
| optimization_with_constraints.ipynb | Practical algorithms and case studies for optimization under constraints. |
| saddle_point.ipynb | Analysis of saddle points in optimization landscapes and their implications. |
| math/convex_functions.ipynb | In-depth look at convex functions, their properties, and relevance to optimization. |

### applications/ADMM/

Applications of ADMM in distributed and large-scale optimization problems.

| Filename | Description |
|----------|-------------|
| ATM.ipynb | Application of ADMM to distributed optimization in ATM networks. |
| Distributed_linear_regression.ipynb | Distributed linear regression using ADMM, with code and results. |
| LogisticRegression.ipynb | Logistic regression solved via ADMM, including implementation details. |
| consensus_cuda.ipynb | GPU-accelerated consensus algorithms using ADMM. |
| sharing_power_grid.ipynb | Power grid sharing optimization formulated and solved with ADMM. |

### graph_consensus/

Consensus algorithms and dynamics on graphs, including average, min, max, and stochastic consensus in various network topologies.

| Filename | Description |
|----------|-------------|
| SchocasticAdjacency.ipynb | Stochastic adjacency matrices and their effect on consensus. |
| connected_graph_avg_min_max_consensus.ipynb | Average, min, and max consensus in connected graphs. |
| distributed_decentrarlized_gradient_descent.ipynb | Distributed decentralized gradient descent algorithms. |
| distributed_decentrarlized_gradient_descent_2nd_order.ipynb | Second-order distributed decentralized gradient descent. |
| fxts_consensus.ipynb | Finite-time consensus algorithms and analysis. |
| laplacian_dynamics.ipynb | Laplacian dynamics and their role in consensus algorithms. |
| uncapacitated_capacitated_edp.ipynb | Uncapacitated and capacitated edge-disjoint paths problems in graph optimization. |
| undirected_unweighted_drone_direction_consensus.ipynb | Drone direction consensus in undirected, unweighted graphs. |
| undirected_weighted_connected_graph_avg_min_max_consensus.ipynb | Consensus in undirected, weighted connected graphs. |

---
For more details, see the individual notebooks. Contributions and suggestions are welcome!