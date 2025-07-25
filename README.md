# Distributed Optimization

## Introduction
This repository contains a collection of Jupyter Notebooks and resources focused on distributed optimization, convex analysis, and consensus algorithms. The materials are organized by topic and cover both foundational concepts and advanced methods, including the Augmented Lagrangian, ADMM, dual ascent, graph-based consensus, and more. The notebooks are suitable for students, researchers, and practitioners interested in optimization, machine learning, and distributed systems.

## Table of Contents

### Augmented Lagrangian & Multipliers
Explores methods for handling constraints in optimization problems using augmented Lagrangian techniques and multiplier methods.
- [AugmentedLagrangian.ipynb](AugmentedLagrangian.ipynb): Introduction and implementation of the Augmented Lagrangian method for constrained optimization, with examples and visualizations.
- [MethodOfMultipliers.ipynb](MethodOfMultipliers.ipynb): In-depth exploration of the method of multipliers, including theory and practical algorithms.

### Dual Ascent
Focuses on the dual ascent algorithm, a foundational approach for distributed optimization and decomposition methods.
- [DualAscent.ipynb](DualAscent.ipynb): Step-by-step development and application of the dual ascent method to distributed problems.

### Neural Network Optimization
Optimization strategies and algorithms for training neural networks efficiently and effectively.
- [NN_Optimization.ipynb](NN_Optimization.ipynb): Covers gradient-based and advanced optimization methods for neural network training.

### Constrained Optimization
Techniques and algorithms for solving optimization problems with constraints, both equality and inequality.
- [constrained_optimization.ipynb](constrained_optimization.ipynb): Methods for formulating and solving constrained optimization problems, with worked examples.
- [optimization_with_constraints.ipynb](optimization_with_constraints.ipynb): Practical algorithms and case studies for optimization under constraints.

### Convergence Analysis
Examines the convergence properties of various optimization algorithms, including rates and conditions for convergence.
- [convergence.ipynb](convergence.ipynb): Theoretical and empirical analysis of convergence in optimization routines.

### Gradient Fields
Visualization and study of gradient fields, including normalized gradients, in the context of optimization.
- [gradient_fields.ipynb](gradient_fields.ipynb): Visual and mathematical exploration of gradient fields in optimization landscapes.
- [normalized_gradient_.ipynb](normalized_gradient_.ipynb): Effects and applications of using normalized gradients in optimization.

### Restricting Optimization Surface
Methods for restricting the optimization process to feasible or desirable regions of the search space.
- [RestrictingOptimizationSurface.ipynb](RestrictingOptimizationSurface.ipynb): Techniques and examples for limiting optimization to specific surfaces or regions.

### Saddle Points & Stability
Understanding the role of saddle points and stability in optimization, including their impact on algorithm performance.
- [saddle_point.ipynb](saddle_point.ipynb): Analysis of saddle points in optimization landscapes and their implications.
- [Stabiilty.ipynb](Stabiilty.ipynb): Stability analysis of optimization algorithms and systems.

### ADMM & Applications
Alternating Direction Method of Multipliers (ADMM) and its use in distributed and large-scale optimization problems.
- [ADMM/ATM.ipynb](ADMM/ATM.ipynb): Application of ADMM to distributed optimization in ATM networks.
- [ADMM/Distributed_linear_regression.ipynb](ADMM/Distributed_linear_regression.ipynb): Distributed linear regression using ADMM, with code and results.
- [ADMM/LogisticRegression.ipynb](ADMM/LogisticRegression.ipynb): Logistic regression solved via ADMM, including implementation details.
- [ADMM/consensus_cuda.ipynb](ADMM/consensus_cuda.ipynb): GPU-accelerated consensus algorithms using ADMM.
- [ADMM/sharing_power_grid.ipynb](ADMM/sharing_power_grid.ipynb): Power grid sharing optimization formulated and solved with ADMM.

### Graph-based Topology & Consensus
Consensus algorithms and dynamics on graphs, including average, min, max, and stochastic consensus in various network topologies.
- [graph-based-topology/connected_graph_avg_min_max_consensus.ipynb](graph-based-topology/connected_graph_avg_min_max_consensus.ipynb): Average, min, and max consensus in connected graphs.
- [graph-based-topology/laplacian_dynamics.ipynb](graph-based-topology/laplacian_dynamics.ipynb): Laplacian dynamics and their role in consensus algorithms.
- [graph-based-topology/SchocasticAdjacency.ipynb](graph-based-topology/SchocasticAdjacency.ipynb): Stochastic adjacency matrices and their effect on consensus.
- [graph-based-topology/undirected_unweighted_drone_direction_consensus.ipynb](graph-based-topology/undirected_unweighted_drone_direction_consensus.ipynb): Drone direction consensus in undirected, unweighted graphs.
- [graph-based-topology/undirected_weighted_connected_graph_avg_min_max_consensus.ipynb](graph-based-topology/undirected_weighted_connected_graph_avg_min_max_consensus.ipynb): Consensus in undirected, weighted connected graphs.

### Convex Functions (Math)
Mathematical properties, definitions, and examples of convex functions, foundational to optimization theory.
- [math/convex_functions.ipynb](math/convex_functions.ipynb): In-depth look at convex functions, their properties, and relevance to optimization.

---
For more details, see the individual notebooks. Contributions and suggestions are welcome!