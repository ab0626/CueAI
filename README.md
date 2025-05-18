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
| Layer                                          | Tool                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| **Language**                                   | Python or C++ (Python for speed of development, C++ for realism/speed) |
| **Physics Engine**                             | Custom or `pymunk` (Python wrapper for Chipmunk2D)                     |
| **Visualization**                              | PyGame / Pyglet (2D), or Web-based (Three.js or WebGL via JS)          |
| **UI**                                         | Streamlit (easy Python GUI) or PyQt (desktop GUI)                      |
| **Computer Vision (optional for calibration)** | OpenCV                                                                 |
| **ML Calibration (optional)**                  | scikit-learn / PyTorch (model to correct predictions)                  |



## ML Stack
| Component     | Tool                                                                   |
| ------------- | ---------------------------------------------------------------------- |
| Data handling | `pandas`, `numpy`                                                      |
| Model type    | Regression (e.g., `RandomForestRegressor`, `XGBoost`, or `Neural Net`) |
| ML Framework  | `scikit-learn` or `PyTorch`                                            |
| Evaluation    | Mean Squared Error / Cosine Similarity                                 |
| Deployment    | Model saved via `joblib` or `onnx` and used during simulation          |

✅ Phase 1: Core Physics Engine
1. Table & Ball Model
Table: standard dimensions, six pockets, rail bounce logic

Balls: mass, radius, moment of inertia

Physics: 2D motion + angular velocity

2. Cue Input System
Shot parameters:

Initial position

Angle (θ)

Speed (v)

Spin: top/back/side (mapped via 2D vector)

3. Ball Behavior Mechanics
Translational movement using Newtonian physics

Rotational movement from spin and torque

Friction modeling:

Sliding → rolling transition

Ball/table and ball/ball

🔁 Phase 2: Advanced Interactions
4. Collision Mechanics
Ball-ball elastic collisions with spin transfer

Ball-rail bounces with spin/speed adjustments

Use impulse and angular momentum conservation

5. Spin Effects
Add logic for:

Squirt: offsetting path due to side spin

Throw: object ball deviates when hit with spin

Masse (curved shots): if cue elevation is supported

6. Curved Table Surface
Simulate using:

Heightmap h(x, y) → slope vector ∇h

Apply drift acceleration based on slope

🧪 Phase 3: Calibration & ML (Optional)
7. Real Table Calibration Mode
Take test shots

Log predicted vs. actual outcomes (via CV)

Train ML model to learn residual errors:

Input: shot parameters

Output: correction offset

👁️ Phase 4: UI + Visualization
8. UI Interface
Drag/drop balls

Adjust cue input (angle, speed, spin)

Toggle effects on/off (e.g., cloth nap, throw)

9. Visual Overlay
Display predicted paths

Show cue stick visualization

Add vector indicators for spin, squirt, etc.
