# config.py

# === Table Constants (meters) ===
TABLE_WIDTH = 2.84     # Standard 9-foot table (in meters)
TABLE_HEIGHT = 1.42
POCKET_RADIUS = 0.06

# === Ball Constants ===
BALL_RADIUS = 0.028575     # Standard ball diameter is 57.15mm
BALL_MASS = 0.17           # In kilograms

# === Physics Constants ===
FRICTION_COEFF = 0.015     # Dynamic friction (table vs ball)
ROLLING_RESISTANCE = 0.002 # Resistance that slows rolling balls
SPIN_DECAY = 0.005         # Spin loss over time

# === Simulation Settings ===
TIME_STEP = 0.01          # Time step for the simulation in seconds
MAX_SIM_TIME = 5.0        # Max duration per shot in seconds
COLLISION_DAMPING = 0.99  # Energy retained after collisions

# === ML Calibration ===
USE_ML_CORRECTION = True
ML_MODEL_PATH = "calibration/ml_model.pkl"
