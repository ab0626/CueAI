# precision-pool-sim
# Precision Pool Simulator (CueAI)

## Overview
CueAI is a high-fidelity pool simulation app designed to predict cue ball and object ball behavior based on input parameters like shot angle, speed, spin, and table imperfections. It incorporates advanced physics simulations, machine learning calibration, and a simple user interface to provide an interactive and accurate pool shot prediction tool.

## Features
- **Physics-based Simulation**: Simulates the behavior of cue balls and object balls, including friction, spin, and collisions.
- **Machine Learning Calibration**: Trains a regression model to adjust predictions based on real-world data, improving accuracy over time.
- **Interactive UI**: Provides a simple drag-and-drop interface for ball placement and shot input, built using PyGame or Streamlit.
- **Real-World Calibration**: Allows users to calibrate the simulation using test shot data and computer vision feedback.

## Tech Stack
Layer	Tool
Language: Python or C++ (Python for speed of development, C++ for realism/speed)
Physics Engine: Custom or pymunk (Python wrapper for Chipmunk2D)
Visualization: PyGame / Pyglet (2D), or Web-based (Three.js or WebGL via JS)
UI:	Streamlit (easy Python GUI) or PyQt (desktop GUI)
Computer Vision: (optional for calibration)	OpenCV
ML Calibration (optional):	scikit-learn / PyTorch (model to correct predictions)


## ML Stack
| Component     | Tool                                                                   |
| ------------- | ---------------------------------------------------------------------- |
| Data handling | `pandas`, `numpy`                                                      |
| Model type    | Regression (e.g., `RandomForestRegressor`, `XGBoost`, or `Neural Net`) |
| ML Framework  | `scikit-learn` or `PyTorch`                                            |
| Evaluation    | Mean Squared Error / Cosine Similarity                                 |
| Deployment    | Model saved via `joblib` or `onnx` and used during simulation          |

