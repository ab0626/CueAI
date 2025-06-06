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

# 🎱 CueAI: High-Fidelity Pool Simulation & Shot Prediction Engine

CueAI is a physics-based, machine learning-enhanced simulator that models pool (billiards) shots with high precision. It predicts cue ball behavior, spin effects, and object ball trajectories using realistic physics, collision mechanics, and optional calibration from real-world data.

---

## 🚀 Features at a Glance

- 2D physics engine with spin and torque modeling  
- Realistic cue input system (angle, spin, velocity)  
- Ball-ball and ball-rail collision handling with spin effects  
- Advanced mechanics like squirt, throw, and masse  
- Curved table simulation via heightmaps  
- Machine learning calibration using computer vision  
- Interactive GUI for shot setup and real-time visualization  

---

## ✅ Phase 1: Core Physics Engine

### Table & Ball Model
- Standard 9-ft table with 6 pockets
- Ball properties: mass, radius, moment of inertia
- Accurate rail bounce geometry

### Cue Input System
- Cue parameters:
  - Initial position (x, y)
  - Shot angle (θ)
  - Speed (v)
  - Spin vector: [top/back, side]

### Physics Modeling
- 2D translational motion via Newtonian equations
- Angular motion: torque and spin propagation
- Friction mechanics:
  - Sliding to rolling transition
  - Ball/table and ball/ball friction handling

---

## 🔁 Phase 2: Advanced Interactions

### Collision Mechanics
- Elastic ball-to-ball collisions with spin transfer
- Ball-to-rail bounces with angular adjustments
- Impulse and angular momentum conservation

### Spin Effects
- **Squirt:** Lateral cue ball deviation due to side spin
- **Throw:** Object ball veers off-line when struck with spin
- **Masse:** Curved trajectories when cue elevation is introduced

### Curved Table Surface
- Modeled with a heightmap `h(x, y)`
- Slope vector ∇h used to apply drift acceleration
- Simulates warped/sloped surfaces for advanced realism

---

## 🧪 Phase 3: Calibration & ML (Optional)

### Real Table Calibration Mode
- Take test shots and log predicted vs. actual ball paths
- Use computer vision (OpenCV) to extract real-world shot data
- Train ML model to correct residual errors

#### ML Model
- **Input:** [cue angle, speed, spin vector]
- **Output:** Offset correction (position, angle)
- Frameworks: PyTorch / Scikit-learn (regressor)

---

## 👁️ Phase 4: UI + Visualization

### Interactive UI
- Drag/drop interface for placing balls
- Cue input controls:
  - Angle slider
  - Speed power bar
  - Spin wheel/vector selector
- Toggle buttons:
  - Enable/disable friction, cloth effects, throw/squirt

### Visual Overlays
- Predicted paths for cue and object balls
- Cue stick aiming guide
- Vector arrows for spin, squirt, throw

Frameworks: PyQt / PyGame / OpenGL (optional)

---

## 🧰 Tech Stack

| Component         | Technology                    |
|------------------|-------------------------------|
| Physics Engine    | Python, NumPy                 |
| UI Layer          | PyQt5 / PyGame                |
| CV Calibration    | OpenCV                        |
| ML Calibration    | PyTorch / Scikit-learn        |
| Visualization     | Matplotlib / OpenGL           |

---

## 📈 Future Enhancements

- Multiplayer shot replay system  
- Shot accuracy analytics dashboard  
- Cue elevation and jump shot mechanics  
- Mobile-friendly version  

---

## 📸 Demo (Coming Soon)

- GIFs of path prediction and spin visualization  
- Before/after of real-world calibration  

---

## 🧠 Credits

Built with passion for physics, machine learning, and precision gaming.

---

## 📄 License

MIT License — use, modify, and share freely.
