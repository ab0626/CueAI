# 🎱 CueAI: Ultra-Focused Pool Video Analyzer

**Advanced AI-powered pool video analysis with single ball calibration and spin prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

CueAI is a revolutionary pool video analysis system that combines computer vision, machine learning, and advanced physics to provide ultra-precise analysis of pool shots. The system can analyze first-person pool videos, detect individual shots, predict spin effects, and provide detailed shot-by-shot breakdowns.

### 🎯 Key Features

- **Ultra-Focused Shot Analysis**: Individual ball-by-ball analysis with precise calibration
- **Spin Prediction**: Advanced algorithms to predict and analyze spin effects
- **Pool Table Detection**: Robust detection of pool tables with multiple color support
- **Strategic Scanning**: Context-aware scanning phases (rack → break → individual shots)
- **Real-Time Processing**: Optimized for performance with smart frame skipping
- **YouTube Integration**: Direct analysis of YouTube pool videos

## 🔬 Technical Architecture

### Core Components

#### 1. **UltraFocusedPoolAnalyzer** (`video_pool_analyzer.py`)
The main analyzer class that implements:
- **Pool table detection** with multiple HSV color ranges (green, blue, brown, red felt)
- **Strategic ball scanning** based on game context
- **Individual shot analysis** with pre-shot, execution, and post-shot phases
- **Spin prediction** using trajectory analysis
- **Performance optimization** with adaptive quality adjustment

#### 2. **Game Context Management**
```python
@dataclass
class GameContext:
    is_initial_rack: bool
    is_post_break: bool
    is_shot_preparation: bool
    is_shot_execution: bool
    current_shot_id: int
```

#### 3. **Shot Analysis System**
```python
@dataclass
class ShotAnalysis:
    shot_id: int
    frame_start: int
    frame_end: int
    cue_ball_start: Tuple[float, float]
    target_ball_start: Tuple[float, float]
    shot_angle: float
    shot_distance: float
    predicted_spin: str
    spin_confidence: float
    shot_outcome: str
    accuracy_score: float
```

### Scanning Phases

1. **Initial Rack Detection** (0-10s): Scans all 15 balls in the rack
2. **Post-Break Scattering** (10-20s): Tracks all balls as they scatter
3. **Individual Shot Analysis** (20s+): 
   - **Standing Phase**: Scans all balls when player is standing
   - **Preparation Phase**: Focuses on target ball when player gets down
   - **Execution Phase**: Tracks shot execution and spin effects

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenCV 4.8+
- NumPy
- yt-dlp

### Setup
```bash
# Clone the repository
git clone https://github.com/ab0626/CueAI.git
cd CueAI

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
opencv-python>=4.8.0
numpy>=1.21.0
yt-dlp>=2023.0.0
matplotlib>=3.5.0
```

## 📖 Usage

### Basic Usage
```python
from video_pool_analyzer import UltraFocusedPoolAnalyzer

# Initialize analyzer
analyzer = UltraFocusedPoolAnalyzer()

# Analyze a YouTube video
video_url = "https://www.youtube.com/watch?v=-fCIN8RQp9s"
analyzer.analyze_video(video_url, start_time=0.32)
```

### Test Script
```bash
# Run the test analyzer
python test_video_analyzer.py "https://www.youtube.com/watch?v=-fCIN8RQp9s"
```

## 🎯 Advanced Features

### 1. Pool Table Detection
- **Multi-color support**: Detects green, blue, brown, and red felt
- **Robust fallback**: Uses full frame if table detection fails
- **Boundary detection**: Identifies table corners and pockets

### 2. Ball Classification
- **Solid balls** (1-7): Identified by color patterns
- **Striped balls** (9-15): Detected by stripe patterns
- **8-ball**: Special classification as solid
- **Cue ball**: Distinguished by size and position

### 3. Spin Prediction
- **Top spin**: Forward rotation effects
- **Bottom spin**: Backward rotation effects
- **Side spin**: Left/right rotation effects
- **Combined spin**: Multiple rotation axes

### 4. Shot Analysis
- **Pre-shot analysis**: Cue ball and target ball positions
- **Execution tracking**: Trajectory and impact detection
- **Post-shot analysis**: Outcome determination and accuracy scoring

## 🔧 Configuration

### Performance Settings
```python
analyzer = UltraFocusedPoolAnalyzer(
    analysis_fps=25,  # Analysis frame rate
    shot_preparation_frames=30,  # Frames to detect shot prep
    shot_execution_frames=60,  # Frames to track execution
    spin_analysis_frames=90  # Frames to analyze spin effects
)
```

### Detection Parameters
- **Ball detection**: HoughCircles with adaptive parameters
- **Table detection**: HSV color ranges with morphological operations
- **Shot detection**: Velocity thresholds for preparation/execution

## 📊 Output

### Visual Output
- **Frame overlays**: Real-time analysis visualization
- **Trajectory paths**: Cue ball and target ball paths
- **Spin indicators**: Visual spin effect markers
- **Shot information**: Angle, distance, predicted spin

### Data Output
- **Shot analyses**: Complete shot-by-shot breakdowns
- **Performance metrics**: Accuracy scores and confidence levels
- **Frame captures**: Key moments saved as images

## 🚀 Performance Optimizations

### Smart Frame Skipping
- Adaptive frame rate based on motion
- Caching of detection results
- Memory-efficient algorithms

### Real-Time Processing
- GPU acceleration support (optional)
- Multi-threaded processing
- Optimized OpenCV operations

## 🎯 Use Cases

### 1. **Player Analysis**
- Shot accuracy assessment
- Spin technique evaluation
- Game strategy analysis

### 2. **Training Tool**
- Visual feedback for players
- Shot breakdown and improvement suggestions
- Performance tracking over time

### 3. **Research Applications**
- Pool physics research
- Computer vision development
- Sports analytics

## 🔬 Technical Details

### Computer Vision Pipeline
1. **Frame preprocessing**: Color space conversion and filtering
2. **Table detection**: HSV segmentation and contour analysis
3. **Ball detection**: HoughCircles with table mask application
4. **Ball classification**: Color analysis and pattern recognition
5. **Trajectory tracking**: Velocity calculation and path prediction

### Machine Learning Integration
- **Spin prediction**: Regression models for spin effects
- **Shot outcome**: Classification models for shot results
- **Performance calibration**: Adaptive learning from real-world data

## 📈 Future Enhancements

### Planned Features
- **Multi-camera support**: Multiple angle analysis
- **Advanced spin detection**: More precise spin measurement
- **Shot recommendation**: AI-powered shot suggestions
- **Mobile app**: iOS/Android companion app
- **Cloud processing**: Web-based analysis platform

### Research Areas
- **3D reconstruction**: Full 3D ball trajectory analysis
- **Advanced physics**: More accurate collision modeling
- **Player identification**: Automatic player recognition
- **Game state tracking**: Complete game progression analysis

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 python/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV Community**: For the excellent computer vision library
- **Pool Physics Research**: For the mathematical foundations
- **YouTube**: For providing the video analysis platform
- **Open Source Community**: For the tools and libraries that made this possible

## 📞 Contact

- **GitHub**: [https://github.com/ab0626/CueAI](https://github.com/ab0626/CueAI)
- **Issues**: [GitHub Issues](https://github.com/ab0626/CueAI/issues)

---

**Built with ❤️ for the pool community**

*CueAI - Where precision meets passion in pool analysis*