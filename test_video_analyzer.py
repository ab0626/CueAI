#!/usr/bin/env python3
"""
Test script for CueAI Ultra-Focused Pool Video Analyzer

Tests the ultra-focused pool analyzer with individual shot analysis and spin prediction.
"""

import sys
import os
from video_pool_analyzer import UltraFocusedPoolAnalyzer

def get_pool_video_urls():
    """Get example pool video URLs for testing."""
    return [
        "https://www.youtube.com/watch?v=-fCIN8RQp9s",  # First-person pool
        "https://www.youtube.com/watch?v=QH2-TGUlwu4",  # Professional pool
        "https://www.youtube.com/watch?v=8jLOx1hD3_o",  # Pool tutorial
    ]

def validate_pool_content(video_source: str) -> bool:
    """Basic validation that this is likely a pool video."""
    # Check for known pool video IDs
    known_pool_videos = [
        '-fCIN8RQp9s',  # First-person pool video
        'QH2-TGUlwu4',  # Professional pool
        '8jLOx1hD3_o',  # Pool tutorial
    ]
    
    # Check if it's a known pool video
    for video_id in known_pool_videos:
        if video_id in video_source:
            return True
    
    # Fallback to keyword checking
    pool_keywords = ['pool', 'billiards', 'snooker', '8-ball', '9-ball', 'cue']
    video_lower = video_source.lower()
    
    for keyword in pool_keywords:
        if keyword in video_lower:
            return True
    
    return False

def main(video_source: str = None):
    """Main test function."""
    if not video_source:
        # Use default pool video
        video_source = "https://www.youtube.com/watch?v=-fCIN8RQp9s"
    
    print("🎱 CueAI Ultra-Focused Pool Video Analyzer Test")
    print("=" * 50)
    
    # Validate video source
    if not validate_pool_content(video_source):
        print(f"Warning: {video_source} may not be a pool video")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Create ultra-focused analyzer with start time
        print(f"Creating ultra-focused analyzer for: {video_source}")
        start_time = 0.32  # Start at 0.32 seconds after break
        analyzer = UltraFocusedPoolAnalyzer(video_source, start_time)
        print(f"Ultra-focused analysis will start at {start_time} seconds (after break)")
        print("Features: Individual shot analysis, spin prediction, ultra-precise tracking")
        
        # Setup video
        print("\nSetting up video...")
        analyzer.setup_video()
        
        # Create output directory
        output_dir = "ultra_focused_test_output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
        
        # Start analysis
        print("\nStarting ultra-focused analysis...")
        print("Press 'q' to quit, 'p' to pause")
        analyzer.analyze_video(output_dir)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 