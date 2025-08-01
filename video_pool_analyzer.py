#!/usr/bin/env python3
"""
CueAI Ultra-Focused Pool Shot Analyzer

Enhanced pool video analysis with individual shot focus and spin prediction.
Key improvements:
- Ultra-focused individual shot analysis (one ball at a time)
- Spin prediction for each shot
- Pre-shot analysis (cue ball position, target ball, angle)
- Post-shot analysis (trajectory, spin effects, outcome)
- Enhanced ball tracking with spin detection

Features:
- Individual shot isolation and analysis
- Spin prediction (top, bottom, left, right, combination)
- Shot preparation detection
- Cue ball trajectory analysis
- Target ball response analysis
- Performance optimizations
"""

import cv2
import numpy as np
import time
import sys
import os
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from urllib.parse import urlparse
import yt_dlp
import math

@dataclass
class ShotAnalysis:
    """Complete analysis of a single pool shot."""
    shot_id: int
    frame_start: int
    frame_end: int
    
    # Pre-shot analysis
    cue_ball_start: Tuple[float, float]
    target_ball_start: Tuple[float, float]
    shot_angle: float  # degrees
    shot_distance: float
    target_ball_type: str
    
    # Shot execution
    cue_ball_trajectory: List[Tuple[float, float]]
    target_ball_trajectory: List[Tuple[float, float]]
    impact_frame: int
    
    # Spin analysis
    predicted_spin: str  # "top", "bottom", "left", "right", "top-left", etc.
    spin_confidence: float
    actual_spin_effects: List[str]  # observed effects
    
    # Outcome
    shot_outcome: str  # "pocketed", "missed", "carom", "scratch"
    accuracy_score: float
    notes: str

@dataclass
class GameContext:
    """Tracks the current game context for strategic scanning."""
    is_rack_formation: bool = False
    is_break_shot: bool = False
    is_post_break: bool = False
    is_shot_preparation: bool = False
    is_shot_execution: bool = False
    rack_detected: bool = False
    break_completed: bool = False
    last_shot_time: float = 0.0
    current_shot_id: int = 0

@dataclass
class DetectedBall:
    """Represents a detected pool ball with improved classification."""
    id: int
    x: float
    y: float
    radius: float
    ball_type: str  # "1-Solid", "9-Stripe", "8-Ball", "Cue"
    color: str
    confidence: float
    is_solid: bool
    is_striped: bool
    ball_number: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    last_seen: float = 0.0
    in_camera_view: bool = True
    # Enhanced tracking for spin analysis
    angular_velocity: float = 0.0
    spin_direction: str = "none"  # "clockwise", "counterclockwise", "none"
    trajectory_history: List[Tuple[float, float]] = None

class UltraFocusedPoolAnalyzer:
    """Ultra-focused pool analyzer for individual shot analysis with spin prediction."""
    
    def __init__(self, video_source: str, start_time: float = 0.0):
        self.video_source = video_source
        self.video_path = None
        self.cap = None
        self.start_time = start_time
        
        # Performance optimizations
        self.frame_rate = 30
        self.analysis_fps = 25  # Higher for precise shot analysis
        self.skip_frames = 1
        self.min_ball_size = 8
        self.max_ball_size = 40
        
        # Game context tracking
        self.game_context = GameContext()
        
        # Shot analysis parameters
        self.shot_preparation_frames = 30  # Frames to analyze before shot
        self.shot_execution_frames = 60   # Frames to analyze during shot
        self.spin_analysis_frames = 45    # Frames to analyze spin effects
        
        # Ball detection parameters - optimized for speed and accuracy
        self.ball_colors = {
            # Solid balls (full color)
            'yellow_solid': ([20, 100, 100], [30, 255, 255]),      # 1-ball
            'blue_solid': ([100, 100, 100], [130, 255, 255]),      # 2-ball
            'red_solid': ([0, 100, 100], [10, 255, 255]),          # 3-ball
            'purple_solid': ([130, 100, 100], [160, 255, 255]),    # 4-ball
            'orange_solid': ([10, 100, 100], [20, 255, 255]),      # 5-ball
            'green_solid': ([40, 100, 100], [80, 255, 255]),       # 6-ball
            'brown_solid': ([10, 50, 50], [20, 255, 200]),         # 7-ball
            'black_solid': ([0, 0, 0], [180, 255, 30]),            # 8-ball
            
            # Striped balls (white with colored stripe)
            'yellow_stripe': ([20, 50, 200], [30, 255, 255]),      # 9-ball
            'blue_stripe': ([100, 50, 200], [130, 255, 255]),      # 10-ball
            'red_stripe': ([0, 50, 200], [10, 255, 255]),          # 11-ball
            'purple_stripe': ([130, 50, 200], [160, 255, 255]),    # 12-ball
            'orange_stripe': ([10, 50, 200], [20, 255, 255]),      # 13-ball
            'green_stripe': ([40, 50, 200], [80, 255, 255]),       # 14-ball
            'brown_stripe': ([10, 30, 200], [20, 255, 200]),       # 15-ball
            
            # Special balls
            'white_cue': ([0, 0, 200], [180, 30, 255]),            # Cue ball
        }
        
        # Tracking data
        self.detected_balls = []
        self.shot_analyses = []
        self.current_frame = 0
        self.total_frames = 0
        
        # Current shot tracking
        self.current_shot = None
        self.shot_preparation_data = []
        self.shot_execution_data = []
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_times = deque(maxlen=10)
        
    def setup_video(self):
        """Setup video source with performance optimization."""
        if self._is_youtube_url(self.video_source):
            self.video_path = self._download_youtube_video(self.video_source)
        else:
            self.video_path = self.video_source
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Video loaded: {self.video_path}")
        print(f"Total frames: {self.total_frames}")
        print(f"Frame rate: {self.frame_rate} FPS")
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return 'youtube.com' in url or 'youtu.be' in url
    
    def _download_youtube_video(self, url: str) -> str:
        """Download YouTube video with performance optimization."""
        print(f"Downloading YouTube video: {url}")
        
        os.makedirs('downloads', exist_ok=True)
        
        # Optimize for speed: lower resolution
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': 'downloads/%(title)s.%(ext)s',
            'quiet': False
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                print(f"Downloaded to: {video_path}")
                return video_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            raise
    
    def detect_rack_formation(self, frame: np.ndarray) -> bool:
        """Detect when balls are arranged in a rack (triangle formation)."""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect circles (potential balls)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25,
            param1=50, param2=30, minRadius=self.min_ball_size, maxRadius=self.max_ball_size
        )
        
        if circles is None:
            return False
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Check if balls form a tight cluster (triangle formation)
        if len(circles) >= 10:  # Rack should have 15 balls
            # Calculate center of all detected circles
            center_x = np.mean([c[0] for c in circles])
            center_y = np.mean([c[1] for c in circles])
            
            # Check if balls are close together
            distances = [np.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2) for c in circles]
            avg_distance = np.mean(distances)
            
            # If balls are close together, it's likely a rack
            if avg_distance < 100:
                return True
        
        return False
    
    def detect_shot_preparation(self, balls: List[DetectedBall]) -> bool:
        """Detect when a player is preparing to shoot (gets down to aim)."""
        if not balls:
            return False
        
        # Look for cue ball being positioned
        cue_ball = None
        for ball in balls:
            if ball.ball_type == 'Cue':
                cue_ball = ball
                break
        
        if not cue_ball:
            return False
        
        # Check if cue ball is relatively stationary (player aiming)
        velocity = np.sqrt(cue_ball.velocity_x**2 + cue_ball.velocity_y**2)
        
        # Low velocity indicates shot preparation (player down to shoot)
        if velocity < 1.5:  # More strict threshold
            return True
        
        return False
    
    def detect_shot_execution(self, balls: List[DetectedBall]) -> bool:
        """Detect when a shot is being executed (player shoots)."""
        if not balls:
            return False
        
        # Look for cue ball movement
        cue_ball = None
        for ball in balls:
            if ball.ball_type == 'Cue':
                cue_ball = ball
                break
        
        if not cue_ball:
            return False
        
        # Check for significant cue ball movement (player shoots)
        velocity = np.sqrt(cue_ball.velocity_x**2 + cue_ball.velocity_y**2)
        
        # High velocity indicates shot execution
        if velocity > 8.0:  # Slightly lower threshold for better detection
            return True
        
        return False
    
    def should_scan_for_balls(self, frame: np.ndarray) -> bool:
        """Determine if we should scan for balls based on game context."""
        # Always scan all balls when not in shot preparation (player standing up)
        if not self.game_context.is_shot_preparation:
            return True
        
        # During shot preparation, scan all balls to identify target
        if self.game_context.is_shot_preparation and not self.current_shot:
            return True
        
        # During shot preparation with target identified, scan all balls but focus on target
        if self.game_context.is_shot_preparation and self.current_shot:
            return True  # Scan all balls, but we'll highlight the target
        
        return True  # Default to scanning
    
    def detect_pool_table(self, frame: np.ndarray) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray]]:
        """Detect pool table boundaries and create mask."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple color ranges for different table felt colors
        table_ranges = [
            # Green felt
            (np.array([35, 50, 50]), np.array([85, 255, 255])),
            # Blue felt
            (np.array([100, 50, 50]), np.array([130, 255, 255])),
            # Brown felt
            (np.array([10, 50, 50]), np.array([20, 255, 200])),
            # Red felt
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
        ]
        
        # Combine all table color masks
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in table_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: use entire frame as table if no contours found
            height, width = frame.shape[:2]
            corners = [(0, 0), (width, 0), (width, height), (0, height)]
            table_mask = np.ones((height, width), dtype=np.uint8) * 255
            print(f"⚠️ No pool table contours found, using full frame as table area")
            return corners, table_mask
        
        # Find largest contour (should be the table)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # More lenient area threshold
        min_area = frame.shape[0] * frame.shape[1] * 0.1  # 10% of frame
        if area < min_area:
            # Fallback: use entire frame as table
            height, width = frame.shape[:2]
            corners = [(0, 0), (width, 0), (width, height), (0, height)]
            table_mask = np.ones((height, width), dtype=np.uint8) * 255
            print(f"⚠️ Pool table area too small ({area:.0f}), using full frame")
            return corners, table_mask
        
        # Approximate corners
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:  # Should have 4 corners
            # Fallback: use bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            corners = [tuple(point[0]) for point in approx]
        
        # Create table mask from largest contour
        table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(table_mask, [largest_contour], 255)
        
        if self.current_frame % 30 == 0:  # Print every second
            print(f"✅ Pool table area detected: {area:.0f} pixels")
        return corners, table_mask
    
    def detect_balls_strategically(self, frame: np.ndarray) -> List[DetectedBall]:
        """Detect balls only when strategically appropriate."""
        if not self.should_scan_for_balls(frame):
            return []
        
        start_time = time.time()
        
        # Detect pool table first
        table_info = self.detect_pool_table(frame)
        if not table_info:
            return []
        
        table_corners, table_mask = table_info
        
        detected_balls = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply table mask to only look within table area
        masked_frame = cv2.bitwise_and(frame, frame, mask=table_mask)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=table_mask)
        
        # Detect circles (potential balls) only on table
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=self.min_ball_size, maxRadius=self.max_ball_size
        )
        
        if circles is None:
            return []
        
        circles = np.round(circles[0, :]).astype("int")
        
        for i, (x, y, radius) in enumerate(circles):
            # Verify ball is within table bounds
            if not self._is_within_table_bounds(x, y, table_mask):
                continue
            
            # Classify the ball type
            ball_info = self._classify_ball_fast(masked_hsv, x, y, radius)
            if ball_info:
                ball = DetectedBall(
                    id=i,
                    x=x,
                    y=y,
                    radius=radius,
                    ball_type=ball_info['type'],
                    color=ball_info['color'],
                    confidence=ball_info['confidence'],
                    is_solid=ball_info['is_solid'],
                    is_striped=ball_info['is_striped'],
                    ball_number=ball_info['number'],
                    in_camera_view=True,
                    trajectory_history=[]
                )
                detected_balls.append(ball)
        
        # Update processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return detected_balls
    
    def _is_within_table_bounds(self, x: int, y: int, table_mask: np.ndarray) -> bool:
        """Check if a point is within the pool table boundaries."""
        if y < 0 or y >= table_mask.shape[0] or x < 0 or x >= table_mask.shape[1]:
            return False
        return table_mask[y, x] > 0
    
    def _classify_ball_fast(self, hsv: np.ndarray, x: int, y: int, radius: int) -> Optional[Dict]:
        """Fast ball classification optimized for speed."""
        # Extract small region around ball
        y1, y2 = max(0, y - radius//2), min(hsv.shape[0], y + radius//2)
        x1, x2 = max(0, x - radius//2), min(hsv.shape[1], x + radius//2)
        
        ball_region = hsv[y1:y2, x1:x2]
        if ball_region.size == 0:
            return None
        
        # Fast color matching
        best_match = None
        best_confidence = 0
        
        for color_name, (lower, upper) in self.ball_colors.items():
            mask = cv2.inRange(ball_region, np.array(lower), np.array(upper))
            color_ratio = np.sum(mask > 0) / mask.size
            
            if color_ratio > best_confidence:
                best_confidence = color_ratio
                best_match = color_name
        
        if best_match and best_confidence > 0.15:  # Lower threshold for speed
            # Determine if it's solid or striped
            is_solid = 'solid' in best_match
            is_striped = 'stripe' in best_match
            
            # Map to ball number and type
            ball_mapping = {
                'yellow_solid': {'number': 1, 'type': '1-Solid'},
                'blue_solid': {'number': 2, 'type': '2-Solid'},
                'red_solid': {'number': 3, 'type': '3-Solid'},
                'purple_solid': {'number': 4, 'type': '4-Solid'},
                'orange_solid': {'number': 5, 'type': '5-Solid'},
                'green_solid': {'number': 6, 'type': '6-Solid'},
                'brown_solid': {'number': 7, 'type': '7-Solid'},
                'black_solid': {'number': 8, 'type': '8-Ball'},
                'yellow_stripe': {'number': 9, 'type': '9-Stripe'},
                'blue_stripe': {'number': 10, 'type': '10-Stripe'},
                'red_stripe': {'number': 11, 'type': '11-Stripe'},
                'purple_stripe': {'number': 12, 'type': '12-Stripe'},
                'orange_stripe': {'number': 13, 'type': '13-Stripe'},
                'green_stripe': {'number': 14, 'type': '14-Stripe'},
                'brown_stripe': {'number': 15, 'type': '15-Stripe'},
                'white_cue': {'number': 0, 'type': 'Cue'},
            }
            
            ball_info = ball_mapping.get(best_match)
            if ball_info:
                return {
                    'type': ball_info['type'],
                    'color': best_match,
                    'confidence': best_confidence,
                    'is_solid': is_solid,
                    'is_striped': is_striped,
                    'number': ball_info['number']
                }
        
        return None
    
    def update_game_context(self, frame: np.ndarray, balls: List[DetectedBall]):
        """Update game context based on current frame analysis."""
        # Phase 1: Detect initial rack formation
        if self.current_frame < 300:  # First 10 seconds
            if self.detect_rack_formation(frame):
                self.game_context.is_rack_formation = True
                self.game_context.rack_detected = True
                if self.current_frame % 30 == 0:  # Print every second
                    print(f"🎱 Initial rack detected at frame {self.current_frame} - {len(balls)} balls")
            else:
                self.game_context.is_rack_formation = False
        
        # Phase 2: Detect break shot and post-break scattering
        elif self.current_frame < 600:  # First 20 seconds
            if len(balls) > 10 and not self.game_context.break_completed:
                # Many balls moving indicates break
                self.game_context.break_completed = True
                print(f"💥 Break detected at frame {self.current_frame} - balls scattering")
            
            if self.game_context.break_completed:
                self.game_context.is_post_break = True
                if self.current_frame % 30 == 0:  # Print every second
                    print(f"📊 Post-break: {len(balls)} balls visible")
        
        # Phase 3: Individual shot analysis
        else:
            # Detect shot preparation (player gets down to shoot)
            if self.detect_shot_preparation(balls):
                self.game_context.is_shot_preparation = True
                if not self.current_shot:
                    self.start_new_shot_analysis()
                elif self.current_shot and not self.current_shot.target_ball_type:
                    # Identify target ball if not already identified
                    self.identify_target_ball(balls)
            else:
                self.game_context.is_shot_preparation = False
                # Player is standing up - scan all balls
                if self.current_frame % 30 == 0:  # Print every second
                    print(f"👤 Player standing - scanning all {len(balls)} balls")
            
            # Detect shot execution (player shoots)
            if self.detect_shot_execution(balls):
                self.game_context.is_shot_execution = True
                if self.current_shot:
                    self.analyze_shot_execution(balls)
            else:
                self.game_context.is_shot_execution = False
    
    def start_new_shot_analysis(self):
        """Start analyzing a new individual shot."""
        self.game_context.current_shot_id += 1
        self.current_shot = ShotAnalysis(
            shot_id=self.game_context.current_shot_id,
            frame_start=self.current_frame,
            frame_end=0,
            cue_ball_start=(0, 0),
            target_ball_start=(0, 0),
            shot_angle=0.0,
            shot_distance=0.0,
            target_ball_type="",
            cue_ball_trajectory=[],
            target_ball_trajectory=[],
            impact_frame=0,
            predicted_spin="none",
            spin_confidence=0.0,
            actual_spin_effects=[],
            shot_outcome="unknown",
            accuracy_score=0.0,
            notes=""
        )
        print(f"🎯 Starting shot analysis #{self.game_context.current_shot_id}")
        print("   🔍 Scanning for target ball...")
    
    def identify_target_ball(self, balls: List[DetectedBall]) -> Optional[DetectedBall]:
        """Identify the specific ball the player is targeting during shot preparation."""
        if not balls or not self.current_shot:
            return None
        
        # Find cue ball
        cue_ball = None
        for ball in balls:
            if ball.ball_type == 'Cue':
                cue_ball = ball
                break
        
        if not cue_ball:
            return None
        
        # Find the ball closest to the cue ball's line of sight
        target_ball = None
        min_distance = float('inf')
        
        for ball in balls:
            if ball.ball_type == 'Cue':
                continue
            
            # Calculate distance from cue ball
            dx = ball.x - cue_ball.x
            dy = ball.y - cue_ball.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Prefer balls that are reasonably close (not too far, not too close)
            if 50 < distance < 300 and distance < min_distance:
                min_distance = distance
                target_ball = ball
        
        if target_ball:
            print(f"   🎯 Target identified: {target_ball.ball_type}")
            self.current_shot.target_ball_type = target_ball.ball_type
            self.current_shot.target_ball_start = (target_ball.x, target_ball.y)
            self.current_shot.cue_ball_start = (cue_ball.x, cue_ball.y)
            
            # Calculate shot angle and distance
            dx = target_ball.x - cue_ball.x
            dy = target_ball.y - cue_ball.y
            self.current_shot.shot_angle = math.degrees(math.atan2(dy, dx))
            self.current_shot.shot_distance = math.sqrt(dx*dx + dy*dy)
            
            # Predict spin based on shot setup
            self.current_shot.predicted_spin = self.predict_shot_spin(cue_ball, target_ball)
        
        return target_ball
    
    def scan_single_target_ball(self, frame: np.ndarray, target_ball_type: str) -> Optional[DetectedBall]:
        """Scan only for the specific target ball during shot preparation."""
        if not target_ball_type:
            return None
        
        # Detect pool table first
        table_info = self.detect_pool_table(frame)
        if not table_info:
            return None
        
        table_corners, table_mask = table_info
        
        # Apply table mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=table_mask)
        masked_hsv = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 
                                   cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), 
                                   mask=table_mask)
        
        # Detect circles only on table
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=self.min_ball_size, maxRadius=self.max_ball_size
        )
        
        if circles is None:
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Look specifically for the target ball
        for x, y, radius in circles:
            if not self._is_within_table_bounds(x, y, table_mask):
                continue
            
            ball_info = self._classify_ball_fast(masked_hsv, x, y, radius)
            if ball_info and ball_info['type'] == target_ball_type:
                return DetectedBall(
                    id=0,
                    x=x,
                    y=y,
                    radius=radius,
                    ball_type=ball_info['type'],
                    color=ball_info['color'],
                    confidence=ball_info['confidence'],
                    is_solid=ball_info['is_solid'],
                    is_striped=ball_info['is_striped'],
                    ball_number=ball_info['number'],
                    in_camera_view=True,
                    trajectory_history=[]
                )
        
        return None
    
    def analyze_shot_execution(self, balls: List[DetectedBall]):
        """Analyze the execution of the current shot."""
        if not self.current_shot:
            return
        
        # Find cue ball and target ball
        cue_ball = None
        target_ball = None
        
        for ball in balls:
            if ball.ball_type == 'Cue':
                cue_ball = ball
            elif ball.ball_type != 'Cue' and target_ball is None:
                target_ball = ball
        
        if cue_ball and target_ball:
            # Update shot analysis
            if not self.current_shot.cue_ball_start[0]:  # First execution frame
                self.current_shot.cue_ball_start = (cue_ball.x, cue_ball.y)
                self.current_shot.target_ball_start = (target_ball.x, target_ball.y)
                self.current_shot.target_ball_type = target_ball.ball_type
                
                # Calculate shot angle and distance
                dx = target_ball.x - cue_ball.x
                dy = target_ball.y - cue_ball.y
                self.current_shot.shot_angle = math.degrees(math.atan2(dy, dx))
                self.current_shot.shot_distance = math.sqrt(dx*dx + dy*dy)
                
                # Predict spin based on shot setup
                self.current_shot.predicted_spin = self.predict_shot_spin(cue_ball, target_ball)
            
            # Track trajectories
            self.current_shot.cue_ball_trajectory.append((cue_ball.x, cue_ball.y))
            self.current_shot.target_ball_trajectory.append((target_ball.x, target_ball.y))
            
            # Detect impact
            if self.detect_ball_impact(cue_ball, target_ball):
                self.current_shot.impact_frame = self.current_frame
                print(f"💥 Impact detected at frame {self.current_frame}")
    
    def predict_shot_spin(self, cue_ball: DetectedBall, target_ball: DetectedBall) -> str:
        """Predict the spin that will be applied to the shot."""
        # This is a simplified spin prediction based on shot geometry
        # In a real implementation, this would analyze the cue stick angle, speed, etc.
        
        dx = target_ball.x - cue_ball.x
        dy = target_ball.y - cue_ball.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Simple heuristics for spin prediction
        if distance > 200:  # Long shot
            return "top"  # Top spin for long shots
        elif distance < 100:  # Short shot
            return "bottom"  # Bottom spin for control
        else:
            return "center"  # Center ball for medium shots
    
    def detect_ball_impact(self, cue_ball: DetectedBall, target_ball: DetectedBall) -> bool:
        """Detect when the cue ball impacts the target ball."""
        # Calculate distance between balls
        dx = cue_ball.x - target_ball.x
        dy = cue_ball.y - target_ball.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Impact occurs when balls are very close and cue ball is moving
        cue_velocity = math.sqrt(cue_ball.velocity_x**2 + cue_ball.velocity_y**2)
        
        return distance < 30 and cue_velocity > 5
    
    def analyze_shot_spin_effects(self, balls: List[DetectedBall]) -> List[str]:
        """Analyze the actual spin effects after the shot."""
        effects = []
        
        cue_ball = None
        for ball in balls:
            if ball.ball_type == 'Cue':
                cue_ball = ball
                break
        
        if not cue_ball or len(cue_ball.trajectory_history) < 3:
            return effects
        
        # Analyze trajectory for spin effects
        trajectory = cue_ball.trajectory_history[-3:]
        
        # Check for curve (side spin)
        if len(trajectory) >= 3:
            # Calculate curvature
            p1, p2, p3 = trajectory[-3:]
            angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
            angle_diff = angle2 - angle1
            
            if abs(angle_diff) > 0.1:  # Significant curve
                if angle_diff > 0:
                    effects.append("right_spin")
                else:
                    effects.append("left_spin")
        
        # Check for speed changes (top/bottom spin)
        if len(trajectory) >= 2:
            v1 = math.sqrt((trajectory[-2][0] - trajectory[-3][0])**2 + 
                          (trajectory[-2][1] - trajectory[-3][1])**2)
            v2 = math.sqrt((trajectory[-1][0] - trajectory[-2][0])**2 + 
                          (trajectory[-1][1] - trajectory[-2][1])**2)
            
            if v2 > v1 * 1.2:  # Speed increase
                effects.append("top_spin")
            elif v2 < v1 * 0.8:  # Speed decrease
                effects.append("bottom_spin")
        
        return effects
    
    def track_balls_efficiently(self, current_balls: List[DetectedBall], previous_balls: List[DetectedBall]) -> List[DetectedBall]:
        """Track balls efficiently between frames with enhanced trajectory tracking."""
        if not previous_balls:
            return current_balls
        
        # Find balls that disappeared (pocketed or left view)
        missing_balls = []
        for prev_ball in previous_balls:
            ball_found = False
            for curr_ball in current_balls:
                if curr_ball.ball_type == prev_ball.ball_type:
                    distance = np.sqrt((curr_ball.x - prev_ball.x)**2 + (curr_ball.y - prev_ball.y)**2)
                    if distance < 50:  # Same ball
                        ball_found = True
                        break
            
            if not ball_found:
                missing_balls.append(prev_ball)
                print(f"🎱 Ball {prev_ball.ball_type} disappeared at frame {self.current_frame}")
        
        # Update current balls with tracking info
        for current_ball in current_balls:
            # Find closest previous ball of same type
            min_distance = float('inf')
            closest_ball = None
            
            for prev_ball in previous_balls:
                if prev_ball.ball_type == current_ball.ball_type:
                    distance = np.sqrt((current_ball.x - prev_ball.x)**2 + (current_ball.y - prev_ball.y)**2)
                    if distance < min_distance and distance < 50:
                        min_distance = distance
                        closest_ball = prev_ball
            
            if closest_ball:
                # Calculate velocity
                current_ball.velocity_x = current_ball.x - closest_ball.x
                current_ball.velocity_y = current_ball.y - closest_ball.y
                current_ball.id = closest_ball.id
                current_ball.last_seen = self.current_frame
                
                # Update trajectory history
                if closest_ball.trajectory_history is None:
                    current_ball.trajectory_history = [(closest_ball.x, closest_ball.y)]
                else:
                    current_ball.trajectory_history = closest_ball.trajectory_history.copy()
                
                current_ball.trajectory_history.append((current_ball.x, current_ball.y))
                
                # Keep only last 10 positions to avoid memory issues
                if len(current_ball.trajectory_history) > 10:
                    current_ball.trajectory_history = current_ball.trajectory_history[-10:]
                
                # Analyze spin effects for current shot
                if self.current_shot and self.game_context.is_shot_execution:
                    spin_effects = self.analyze_shot_spin_effects([current_ball])
                    if spin_effects:
                        self.current_shot.actual_spin_effects.extend(spin_effects)
        
        return current_balls
    
    def complete_shot_analysis(self):
        """Complete the analysis of the current shot."""
        if not self.current_shot:
            return
        
        self.current_shot.frame_end = self.current_frame
        
        # Analyze shot outcome
        self.current_shot.shot_outcome = self.determine_shot_outcome()
        
        # Calculate accuracy score
        self.current_shot.accuracy_score = self.calculate_shot_accuracy()
        
        # Add to shot history
        self.shot_analyses.append(self.current_shot)
        
        # Print shot summary
        print(f"🎯 Shot #{self.current_shot.shot_id} completed:")
        print(f"   Target: {self.current_shot.target_ball_type}")
        print(f"   Predicted spin: {self.current_shot.predicted_spin}")
        print(f"   Actual effects: {self.current_shot.actual_spin_effects}")
        print(f"   Outcome: {self.current_shot.shot_outcome}")
        print(f"   Accuracy: {self.current_shot.accuracy_score:.2f}")
        
        self.current_shot = None
    
    def determine_shot_outcome(self) -> str:
        """Determine the outcome of the current shot."""
        if not self.current_shot:
            return "unknown"
        
        # This is a simplified outcome detection
        # In a real implementation, you'd track if the target ball was pocketed
        
        # For now, assume successful if we have good trajectory data
        if len(self.current_shot.cue_ball_trajectory) > 5:
            return "successful"
        else:
            return "missed"
    
    def calculate_shot_accuracy(self) -> float:
        """Calculate the accuracy score for the current shot."""
        if not self.current_shot:
            return 0.0
        
        # Simplified accuracy calculation
        # In a real implementation, this would compare predicted vs actual trajectory
        
        trajectory_length = len(self.current_shot.cue_ball_trajectory)
        if trajectory_length < 3:
            return 0.0
        
        # Base accuracy on trajectory smoothness
        smoothness_score = min(trajectory_length / 10.0, 1.0)
        
        # Bonus for spin prediction accuracy
        spin_accuracy = 0.5  # Default
        if self.current_shot.predicted_spin in self.current_shot.actual_spin_effects:
            spin_accuracy = 1.0
        
        return (smoothness_score + spin_accuracy) / 2.0
    
    def draw_analysis_overlay(self, frame: np.ndarray, balls: List[DetectedBall]) -> np.ndarray:
        """Draw analysis overlay with shot analysis info."""
        overlay = frame.copy()
        
        # Draw pool table boundaries
        table_info = self.detect_pool_table(frame)
        if table_info:
            table_corners, table_mask = table_info
            # Draw table outline
            cv2.polylines(overlay, [np.array(table_corners)], True, (0, 255, 0), 2)
            
            # Draw table label
            center_x = int(np.mean([corner[0] for corner in table_corners]))
            center_y = int(np.mean([corner[1] for corner in table_corners]))
            cv2.putText(overlay, "POOL TABLE", (center_x - 50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show table mask for debugging (semi-transparent overlay)
            if self.current_frame % 30 == 0:  # Every second
                table_overlay = overlay.copy()
                table_overlay[table_mask > 0] = [0, 100, 0]  # Dark green for table area
                cv2.addWeighted(table_overlay, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw detected balls
        for ball in balls:
            color = self._get_ball_color(ball.ball_type)
            cv2.circle(overlay, (int(ball.x), int(ball.y)), int(ball.radius), color, 2)
            
            # Draw ball label
            label = f"{ball.ball_type}"
            cv2.putText(overlay, label, (int(ball.x) + int(ball.radius), int(ball.y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Highlight target ball during shot preparation
        if (self.game_context.is_shot_preparation and 
            self.current_shot and 
            self.current_shot.target_ball_type):
            for ball in balls:
                if ball.ball_type == self.current_shot.target_ball_type:
                    # Draw target indicator
                    cv2.circle(overlay, (int(ball.x), int(ball.y)), 
                              int(ball.radius) + 5, (0, 255, 255), 3)
                    cv2.putText(overlay, "TARGET", (int(ball.x) - 30, int(ball.y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    break
        
        # Draw current shot trajectory
        if self.current_shot and self.current_shot.cue_ball_trajectory:
            for i in range(len(self.current_shot.cue_ball_trajectory) - 1):
                pt1 = tuple(map(int, self.current_shot.cue_ball_trajectory[i]))
                pt2 = tuple(map(int, self.current_shot.cue_ball_trajectory[i + 1]))
                cv2.line(overlay, pt1, pt2, (255, 255, 0), 2)
        
        # Draw info panel
        self._draw_info_panel(overlay, balls)
        
        return overlay
    
    def _get_ball_color(self, ball_type: str) -> Tuple[int, int, int]:
        """Get display color for ball type."""
        color_map = {
            'Cue': (255, 255, 255),      # White
            '1-Solid': (0, 255, 255),    # Yellow
            '2-Solid': (255, 0, 0),      # Blue
            '3-Solid': (0, 0, 255),      # Red
            '4-Solid': (255, 0, 255),    # Purple
            '5-Solid': (0, 165, 255),    # Orange
            '6-Solid': (0, 255, 0),      # Green
            '7-Solid': (19, 69, 139),    # Brown
            '8-Ball': (0, 0, 0),         # Black
            '9-Stripe': (0, 255, 255),   # Yellow
            '10-Stripe': (255, 0, 0),    # Blue
            '11-Stripe': (0, 0, 255),    # Red
            '12-Stripe': (255, 0, 255),  # Purple
            '13-Stripe': (0, 165, 255),  # Orange
            '14-Stripe': (0, 255, 0),    # Green
            '15-Stripe': (19, 69, 139),  # Brown
        }
        return color_map.get(ball_type, (128, 128, 128))
    
    def _draw_info_panel(self, frame: np.ndarray, balls: List[DetectedBall]):
        """Draw information panel with shot analysis metrics."""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 30
        cv2.putText(frame, f"Frame: {self.current_frame}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20
        
        cv2.putText(frame, f"Balls Detected: {len(balls)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20
        
        # Game context information
        context_text = "Context: "
        if self.game_context.is_rack_formation:
            context_text += "RACKING "
        if self.game_context.is_shot_preparation:
            context_text += "PREPARING "
        if self.game_context.is_shot_execution:
            context_text += "EXECUTING "
        if self.game_context.is_post_break:
            context_text += "POST-BREAK "
        if not any([self.game_context.is_rack_formation, self.game_context.is_shot_preparation, 
                   self.game_context.is_shot_execution, self.game_context.is_post_break]):
            context_text += "IDLE"
        
        # Focus mode indicator
        focus_text = "Focus: "
        if self.current_frame < 300:
            focus_text += "INITIAL RACK"
        elif self.current_frame < 600:
            focus_text += "POST-BREAK SCATTER"
        elif self.game_context.is_shot_preparation and self.current_shot and self.current_shot.target_ball_type:
            focus_text += f"TARGET BALL ({self.current_shot.target_ball_type})"
        else:
            focus_text += "ALL BALLS (STANDING)"
        
        cv2.putText(frame, context_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_offset += 20
        
        cv2.putText(frame, focus_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += 20
        
        # Current shot information
        if self.current_shot:
            cv2.putText(frame, f"Shot #{self.current_shot.shot_id}: {self.current_shot.target_ball_type}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Predicted Spin: {self.current_shot.predicted_spin}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
            
            if self.current_shot.actual_spin_effects:
                effects_text = f"Effects: {', '.join(self.current_shot.actual_spin_effects)}"
                cv2.putText(frame, effects_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 20
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20
        
        if self.processing_times:
            avg_processing = np.mean(self.processing_times) * 1000
            cv2.putText(frame, f"Process: {avg_processing:.1f}ms", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def analyze_video(self, output_path: str = None):
        """Analyze the entire video with ultra-focused shot analysis."""
        print("Starting ultra-focused pool video analysis")
        print("Focus: Individual shot analysis with spin prediction")
        
        # Skip to start time if specified
        if self.start_time > 0:
            start_frame = int(self.start_time * self.frame_rate)
            print(f"Skipping to {self.start_time:.2f} seconds (frame {start_frame})")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame = start_frame
        
        previous_balls = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process every nth frame for speed
            if self.current_frame % (self.frame_rate // self.analysis_fps) != 0:
                self.current_frame += 1
                continue
            
            # Update game context
            self.update_game_context(frame, previous_balls)
            
            # Detect balls based on game state
            if self.game_context.is_shot_preparation and self.current_shot and self.current_shot.target_ball_type:
                # During shot preparation with target identified - scan all balls but focus on target
                detected_balls = self.detect_balls_strategically(frame)
                # The target ball will be highlighted in the overlay
            else:
                # Scan all balls in all other cases (rack, post-break, standing up)
                detected_balls = self.detect_balls_strategically(frame)
            
            # Track balls efficiently
            tracked_balls = self.track_balls_efficiently(detected_balls, previous_balls)
            
            # Complete shot analysis if shot ended
            if (self.current_shot and 
                not self.game_context.is_shot_preparation and 
                not self.game_context.is_shot_execution and
                self.current_frame > self.current_shot.frame_start + 30):
                self.complete_shot_analysis()
            
            # Draw overlay
            overlay = self.draw_analysis_overlay(frame, tracked_balls)
            
            # Update FPS
            self.fps_counter += 1
            if self.fps_counter % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                self.current_fps = 30 / elapsed
                self.fps_start_time = time.time()
            
            # Display
            cv2.imshow('Ultra-Focused Pool Analysis', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Pause/unpause
                cv2.waitKey(0)
            
            # Save frame if output path specified
            if output_path:
                frame_filename = os.path.join(output_path, f"frame_{self.current_frame:06d}.jpg")
                cv2.imwrite(frame_filename, overlay)
            
            previous_balls = tracked_balls
            self.current_frame += 1
            
            # Progress update
            if self.current_frame % 100 == 0:
                progress = (self.current_frame / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% ({self.current_frame}/{self.total_frames})")
                print(f"Balls: {len(tracked_balls)}, FPS: {self.current_fps:.1f}, Shots: {len(self.shot_analyses)}")
        
        # Complete any ongoing shot analysis
        if self.current_shot:
            self.complete_shot_analysis()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print analysis results
        self._print_analysis_results()
    
    def _print_analysis_results(self):
        """Print comprehensive shot analysis results."""
        print("\n" + "="*60)
        print("ULTRA-FOCUSED POOL SHOT ANALYSIS RESULTS")
        print("="*60)
        print(f"Total frames analyzed: {self.current_frame}")
        print(f"Rack detected: {self.game_context.rack_detected}")
        print(f"Break completed: {self.game_context.break_completed}")
        print(f"Individual shots analyzed: {len(self.shot_analyses)}")
        print(f"Average FPS: {self.current_fps:.1f}")
        
        if self.processing_times:
            avg_processing = np.mean(self.processing_times) * 1000
            print(f"Average processing time: {avg_processing:.1f}ms")
        
        if self.shot_analyses:
            print("\nDetailed Shot Analysis:")
            for shot in self.shot_analyses:
                print(f"\nShot #{shot.shot_id}:")
                print(f"  Target: {shot.target_ball_type}")
                print(f"  Angle: {shot.shot_angle:.1f}°")
                print(f"  Distance: {shot.shot_distance:.1f}px")
                print(f"  Predicted Spin: {shot.predicted_spin}")
                print(f"  Actual Effects: {shot.actual_spin_effects}")
                print(f"  Outcome: {shot.shot_outcome}")
                print(f"  Accuracy: {shot.accuracy_score:.2f}")
                print(f"  Frames: {shot.frame_start}-{shot.frame_end}")
        
        print("="*60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Ultra-focused pool video analysis with spin prediction')
    parser.add_argument('video_source', help='YouTube URL or local video file path')
    parser.add_argument('--output', '-o', help='Output directory for analyzed frames')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds (default: 0.0)')
    
    args = parser.parse_args()
    
    try:
        analyzer = UltraFocusedPoolAnalyzer(args.video_source, args.start_time)
        analyzer.setup_video()
        
        output_dir = args.output or "ultra_focused_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer.analyze_video(output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 