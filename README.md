# WENO-CFD-Solver

This repository is for storing codes in different stage to develop a 2D/3D compressible Euler equation by application of WENO and FVM.

The project is seperated into several stages:
- 1D Burger's equation
- 1D Euler equation
- 2D Euler equation
- 2D Euler Equation with IBM
- 2D Euler Equation with IBM and AMR

## Introduction
This repository implements the Weighted Essentially Non-Oscillatory (WENO) algorithm. The WENO method is designed to handle discontinuities and sharp gradients in a solution while providing high-order accuracy.

## Methodology
The methodology revolves around the use of the Finite Volume Method (FVM) to approximate the conservation laws. The FVM approach ensures the conservation properties are preserved over the computational domain by integrating the governing equations over control volumes.

### WENO Algorithm
The WENO algorithm is used to obtain high-order accurate numerical solutions. It achieves this by blending multiple polynomial reconstructions of the solution, dynamically adjusting the weights based on the smoothness of the solution. This results in improved resolution of shock waves and discontinuities compared to traditional methods.

### HLLC Riemann Solver
The Harten-Lax-van Leer Contact (HLLC) Riemann solver is applied to resolve the small-scale features of the solution. It provides a way to compute numerical fluxes at the interfaces between computational cells, handling the characteristic propagation of waves in the solution. The HLLC solver is able to deal with both smooth and discontinuous flow conditions effectively.

## Implementations
Within this repository, various test cases for Riemann problems have been implemented to validate the accuracy and efficiency of the WENO method in conjunction with the HLLC Riemann solver. Each test case evaluates the performance under different initial conditions and shock strengths, showcasing the robustness of the solver.

The repository structure and file descriptions are provided for user guidance.