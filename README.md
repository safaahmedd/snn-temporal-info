# SNN Temporal Information for Gesture Recognition

## Authors
Safa Ahmed

## Research Question
How does the number of temporal frames (T) affect the accuracy,
computational cost, and robustness of SNN-based gesture recognition?

## Project Overview
This project investigates how much temporal information is required
for reliable event-based hand gesture recognition using Spiking
Neural Networks (SNNs).

Event-camera recordings are represented as temporal frames with
two polarity channels and a spatial resolution of 48 x 48.

## Experimental Settings
- Window duration: 1.3 seconds
- Spatial resolution: 48 x 48
- Frame representation: Binary frames
- Training epochs: 20
- Random seeds: 0, 1, 2

## Repository Structure
- src/: Python source code
- configs/: Experiment configurations
- notebooks/: Jupyter notebooks
- results/: CSV experiment results
- figures/: Generated graphs

## Dataset
Dataset: DVS128 Gesture (confirm the exact source/version)

Training samples: 1176
Testing samples: 288

Large datasets, processed caches, and model checkpoints are
stored separately in Google Drive and are not included in this repository.

## Setup
Install dependencies:

    pip install -r requirements.txt

## Running the Project
Run the relevant source scripts from the project environment.

Build the processed cache using the build_cache workflow.

Run experiments using the sweep workflow.

Refer to the source code and notebook for the exact commands and arguments.

## Results
Experiment results and visualisations are available in the
results/ and figures/ folders.
