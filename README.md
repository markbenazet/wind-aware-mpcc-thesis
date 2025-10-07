# Wind-Aware MPCC for Fixed-Wing UAV Guidance

**Bachelor Thesis** | ETH Zurich | Autonomous Systems Lab | Fall 2024

## Overview

This thesis develops a **Model Predictive Contouring Control (MPCC)** framework for fixed-wing UAV path following in high wind conditions. Unlike traditional methods (e.g., L1 guidance), MPCC can:

- Handle winds up to 20 m/s while maintaining path accuracy
- Enforce strict safety constraints (acceleration, bank angle limits)
- Adaptively slow down or move backward in extreme tailwinds

**Supervisors**: David Rohr, Dr. Andrea Carron  
**Professor**: Prof. Dr. Roland Siegwart

**Key Innovation**: Separates geometric path following from temporal progression, allowing the controller to dynamically balance safety, path accuracy, and forward progress.
