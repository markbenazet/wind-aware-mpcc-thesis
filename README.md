# MPCC (Model Predictive Contouring Control) for Fixed-Wing UAV

## Overview
This project implements a Model Predictive Contouring Control (MPCC) algorithm for a fixed-wing UAV. It aims to enable the UAV to follow predefined paths while considering various constraints and optimizing its trajectory.

## Features
- Path following for fixed-wing UAV
- Wind disturbance consideration
- Visualization of UAV trajectory and states

## Requirements
- Python 3.x
- CasADi
- NumPy
- SciPy
- Matplotlib
- Acados

## Structure
- `main.py`: Main execution script
- `acados_settings.py`: Acados OCP solver configuration
- `FW_lateral_model.py`: Fixed-wing UAV model definition
- `curve.py`: Path generation and management
- `utils.py`: Utility functions for visualization and data processing
