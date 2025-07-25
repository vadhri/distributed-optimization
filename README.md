# Distributed Optimization

## Introduction
This repository contains a collection of Jupyter Notebooks and resources focused on distributed optimization, convex analysis, and consensus algorithms. The materials are organized by topic and cover both foundational concepts and advanced methods, including the Augmented Lagrangian, ADMM, dual ascent, graph-based consensus, and more. The files only contains python code that can be executed and theory part. 

## Table of Contents


## Organization

### core_methods/
Foundational optimization methods, theory, and mathematical background.

- [core_methods/AugmentedLagrangian.ipynb](core_methods/AugmentedLagrangian.ipynb): Introduction and implementation of the Augmented Lagrangian method for constrained optimization, with examples and visualizations.
- [core_methods/MethodOfMultipliers.ipynb](core_methods/MethodOfMultipliers.ipynb): In-depth exploration of the method of multipliers, including theory and practical algorithms.
- [core_methods/DualAscent.ipynb](core_methods/DualAscent.ipynb): Step-by-step development and application of the dual ascent method to distributed problems.
- [core_methods/NN_Optimization.ipynb](core_methods/NN_Optimization.ipynb): Covers gradient-based and advanced optimization methods for neural network training.
- [core_methods/constrained_optimization.ipynb](core_methods/constrained_optimization.ipynb): Methods for formulating and solving constrained optimization problems, with worked examples.
- [core_methods/optimization_with_constraints.ipynb](core_methods/optimization_with_constraints.ipynb): Practical algorithms and case studies for optimization under constraints.
- [core_methods/convergence.ipynb](core_methods/convergence.ipynb): Theoretical and empirical analysis of convergence in optimization routines.
- [core_methods/gradient_fields.ipynb](core_methods/gradient_fields.ipynb): Visual and mathematical exploration of gradient fields in optimization landscapes.
- [core_methods/normalized_gradient_.ipynb](core_methods/normalized_gradient_.ipynb): Effects and applications of using normalized gradients in optimization.
- [core_methods/RestrictingOptimizationSurface.ipynb](core_methods/RestrictingOptimizationSurface.ipynb): Techniques and examples for limiting optimization to specific surfaces or regions.
- [core_methods/saddle_point.ipynb](core_methods/saddle_point.ipynb): Analysis of saddle points in optimization landscapes and their implications.
- [core_methods/Stabiilty.ipynb](core_methods/Stabiilty.ipynb): Stability analysis of optimization algorithms and systems.
- [core_methods/math/convex_functions.ipynb](core_methods/math/convex_functions.ipynb): In-depth look at convex functions, their properties, and relevance to optimization.

### applications/ADMM/
Applications of ADMM in distributed and large-scale optimization problems.

- [applications/ADMM/ATM.ipynb](applications/ADMM/ATM.ipynb): Application of ADMM to distributed optimization in ATM networks.
- [applications/ADMM/Distributed_linear_regression.ipynb](applications/ADMM/Distributed_linear_regression.ipynb): Distributed linear regression using ADMM, with code and results.
- [applications/ADMM/LogisticRegression.ipynb](applications/ADMM/LogisticRegression.ipynb): Logistic regression solved via ADMM, including implementation details.
- [applications/ADMM/consensus_cuda.ipynb](applications/ADMM/consensus_cuda.ipynb): GPU-accelerated consensus algorithms using ADMM.
- [applications/ADMM/sharing_power_grid.ipynb](applications/ADMM/sharing_power_grid.ipynb): Power grid sharing optimization formulated and solved with ADMM.

### graph_consensus/
Consensus algorithms and dynamics on graphs, including average, min, max, and stochastic consensus in various network topologies.

- [graph_consensus/connected_graph_avg_min_max_consensus.ipynb](graph_consensus/connected_graph_avg_min_max_consensus.ipynb): Average, min, and max consensus in connected graphs.
- [graph_consensus/laplacian_dynamics.ipynb](graph_consensus/laplacian_dynamics.ipynb): Laplacian dynamics and their role in consensus algorithms.
- [graph_consensus/SchocasticAdjacency.ipynb](graph_consensus/SchocasticAdjacency.ipynb): Stochastic adjacency matrices and their effect on consensus.
- [graph_consensus/undirected_unweighted_drone_direction_consensus.ipynb](graph_consensus/undirected_unweighted_drone_direction_consensus.ipynb): Drone direction consensus in undirected, unweighted graphs.
- [graph_consensus/undirected_weighted_connected_graph_avg_min_max_consensus.ipynb](graph_consensus/undirected_weighted_connected_graph_avg_min_max_consensus.ipynb): Consensus in undirected, weighted connected graphs.

---
For more details, see the individual notebooks. Contributions and suggestions are welcome!