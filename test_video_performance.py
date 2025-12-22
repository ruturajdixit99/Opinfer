"""
Video Performance Testing Script for Opinfer

This script tests Opinfer's motion-gated inference on real-world video scenarios
and generates comprehensive performance visualizations.

Usage:
    python test_video_performance.py

Requirements:
    - Videos should be placed in the 'vdo/' folder
    - Supported formats: MP4, AVI, MOV
    - GPU recommended for faster processing

Output:
    - Performance graphs saved to 'graphs/' folder
    - Detailed analysis showing:
      * Motion scores over time
      * Detection accuracy
      * Model call frequency (skip rate)
      * Performance improvements
"""

import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from opinfer import OptimizedInference, load_video_frames
from opinfer.models import ModelLoader

# Global data collection
detection_data = {
    'frame_numbers': [],
    'motion_scores': [],
    'detection_counts': [],
    'max_scores': [],
    'avg_scores': [],
    'object_types': defaultdict(int),
    'model_calls': [],
    'inference_times': [],
    'all_detections': [],  # List of (frame_num, object_name, score)
}


def collect_detection_data(frame_num: int, stats: Dict, output: Dict, text_queries: List[str]):
    """Collect detection data for visualization."""
    detection_data['frame_numbers'].append(frame_num)
    detection_data['motion_scores'].append(stats.get('motion_score', 0))
    detection_data['model_calls'].append(1 if stats.get('did_call', False) else 0)
    detection_data['inference_times'].append(stats.get('inference_ms', 0))
    
    if output is not None and isinstance(output, dict) and 'scores' in output:
        scores = output['scores']
        labels = output['labels']
        
        if len(scores) > 0:
            detection_data['detection_counts'].append(len(scores))
            detection_data['max_scores'].append(float(scores.max()))
            detection_data['avg_scores'].append(float(scores.mean()))
            
            # Collect object type counts
            for label_id, score in zip(labels, scores):
                label_id_int = int(label_id)
                if 0 <= label_id_int < len(text_queries):
                    obj_name = text_queries[label_id_int]
                    detection_data['object_types'][obj_name] += 1
                    detection_data['all_detections'].append((frame_num, obj_name, float(score)))
        else:
            detection_data['detection_counts'].append(0)
            detection_data['max_scores'].append(0)
            detection_data['avg_scores'].append(0)
    else:
        detection_data['detection_counts'].append(0)
        detection_data['max_scores'].append(0)
        detection_data['avg_scores'].append(0)


def generate_graphs(video_name: str, output_dir: str = "graphs"):
    """Generate all visualization graphs."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Detection Scores Over Time
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(detection_data['frame_numbers'], detection_data['max_scores'], 'b-', alpha=0.7, label='Max Score')
    ax1.plot(detection_data['frame_numbers'], detection_data['avg_scores'], 'g-', alpha=0.7, label='Avg Score')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Detection Score')
    ax1.set_title('Detection Scores Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Number of Detections Per Frame
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(detection_data['frame_numbers'], detection_data['detection_counts'], width=1.0, alpha=0.6, color='orange')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Number of Detections')
    ax2.set_title('Detection Count Per Frame')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Object Type Distribution
    ax3 = plt.subplot(3, 3, 3)
    if detection_data['object_types']:
        obj_names = list(detection_data['object_types'].keys())
        obj_counts = list(detection_data['object_types'].values())
        colors = plt.cm.Set3(range(len(obj_names)))
        ax3.barh(obj_names, obj_counts, color=colors)
        ax3.set_xlabel('Total Detections')
        ax3.set_title('Object Type Distribution')
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Motion Scores Over Time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(detection_data['frame_numbers'], detection_data['motion_scores'], 'r-', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Motion Score')
    ax4.set_title('Motion Scores Over Time')
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Call Frequency
    ax5 = plt.subplot(3, 3, 5)
    model_call_frames = [f for f, call in zip(detection_data['frame_numbers'], detection_data['model_calls']) if call == 1]
    ax5.bar(model_call_frames, [1] * len(model_call_frames), width=1.0, alpha=0.6, color='green')
    ax5.set_xlabel('Frame Number')
    ax5.set_ylabel('Model Called (1=Yes, 0=No)')
    ax5.set_title('Model Call Frequency (Green = Model Called)')
    ax5.set_ylim(0, 1.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Inference Time Per Frame
    ax6 = plt.subplot(3, 3, 6)
    inference_frames = [f for f, time_ms in zip(detection_data['frame_numbers'], detection_data['inference_times']) if time_ms > 0]
    inference_times = [t for t in detection_data['inference_times'] if t > 0]
    if inference_times:
        ax6.bar(inference_frames, inference_times, width=1.0, alpha=0.6, color='purple')
        ax6.set_xlabel('Frame Number')
        ax6.set_ylabel('Inference Time (ms)')
        ax6.set_title('Inference Time Per Model Call')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Motion Score vs Detection Count Scatter
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(detection_data['motion_scores'], detection_data['detection_counts'], 
                alpha=0.5, c=detection_data['max_scores'], cmap='viridis', s=30)
    ax7.set_xlabel('Motion Score')
    ax7.set_ylabel('Detection Count')
    ax7.set_title('Motion Score vs Detection Count')
    ax7.grid(True, alpha=0.3)
    if len(detection_data['motion_scores']) > 0:
        plt.colorbar(ax7.collections[0], ax=ax7, label='Max Detection Score')
    
    # 8. Cumulative Model Calls
    ax8 = plt.subplot(3, 3, 8)
    cumulative_calls = np.cumsum(detection_data['model_calls'])
    ax8.plot(detection_data['frame_numbers'], cumulative_calls, 'b-', linewidth=2)
    ax8.set_xlabel('Frame Number')
    ax8.set_ylabel('Cumulative Model Calls')
    ax8.set_title('Cumulative Model Calls Over Time')
    ax8.grid(True, alpha=0.3)
    
    # 9. Score Distribution Histogram
    ax9 = plt.subplot(3, 3, 9)
    all_scores = [score for _, _, score in detection_data['all_detections']]
    if all_scores:
        ax9.hist(all_scores, bins=50, alpha=0.7, color='teal', edgecolor='black')
        ax9.set_xlabel('Detection Score')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Distribution of Detection Scores')
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Opinfer Performance Analysis: {video_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f"{video_name}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Comprehensive analysis graph saved to: {output_file}")
    
    # Also save individual graphs
    save_individual_graphs(video_name, output_path)
    
    plt.close()


def save_individual_graphs(video_name: str, output_dir: Path):
    """Save individual high-quality graphs."""
    # Detection scores over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(detection_data['frame_numbers'], detection_data['max_scores'], 'b-', alpha=0.7, label='Max Score', linewidth=2)
    ax.plot(detection_data['frame_numbers'], detection_data['avg_scores'], 'g-', alpha=0.7, label='Avg Score', linewidth=2)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Detection Score', fontsize=12)
    ax.set_title(f'Detection Scores Over Time - {video_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_scores.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Object distribution
    if detection_data['object_types']:
        fig, ax = plt.subplots(figsize=(10, 6))
        obj_names = list(detection_data['object_types'].keys())
        obj_counts = list(detection_data['object_types'].values())
        colors = plt.cm.Set3(range(len(obj_names)))
        bars = ax.barh(obj_names, obj_counts, color=colors)
        ax.set_xlabel('Total Detections', fontsize=12)
        ax.set_title(f'Object Type Distribution - {video_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, obj_counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / f"{video_name}_objects.png", dpi=200, bbox_inches='tight')
        plt.close()


def calculate_performance_metrics() -> Dict:
    """Calculate and return performance metrics."""
    total_frames = len(detection_data['frame_numbers'])
    total_model_calls = sum(detection_data['model_calls'])
    skip_rate = (1 - (total_model_calls / total_frames)) * 100 if total_frames > 0 else 0
    
    avg_motion_score = np.mean(detection_data['motion_scores']) if detection_data['motion_scores'] else 0
    avg_inference_time = np.mean([t for t in detection_data['inference_times'] if t > 0]) if detection_data['inference_times'] else 0
    
    total_detections = len(detection_data['all_detections'])
    avg_detection_score = np.mean([score for _, _, score in detection_data['all_detections']]) if detection_data['all_detections'] else 0
    
    return {
        'total_frames': total_frames,
        'total_model_calls': total_model_calls,
        'skip_rate_pct': skip_rate,
        'avg_motion_score': avg_motion_score,
        'avg_inference_time_ms': avg_inference_time,
        'total_detections': total_detections,
        'avg_detection_score': avg_detection_score,
        'object_types_detected': len(detection_data['object_types']),
    }


def process_video_with_data_collection(video_path: str, video_name: str, max_frames: int = 500):
    """Process video and collect detection data."""
    # Reset data
    global detection_data
    detection_data = {
        'frame_numbers': [],
        'motion_scores': [],
        'detection_counts': [],
        'max_scores': [],
        'avg_scores': [],
        'object_types': defaultdict(int),
        'model_calls': [],
        'inference_times': [],
        'all_detections': [],
    }
    
    # Load model
    model_name = "owlvit-base"
    model_type = "detector"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detection_queries = ["person", "car", "road", "building", "tree", "traffic light", "sky"]
    
    print(f"\n[INFO] Loading model: {model_name} on {device}...")
    model, processor = ModelLoader.load_detector(model_name, device=device)
    
    # Initialize opinfer
    opinfer = OptimizedInference(
        model_name=model_name,
        model_type=model_type,
        device=device,
        auto_optimize=True,  # Enable automatic optimization
        target_skip_rate=40.0,
    )
    
    # Load frames
    print(f"[INFO] Loading video: {video_path}")
    frames = load_video_frames(video_path, max_frames=max_frames)
    print(f"   [OK] Loaded {len(frames)} frames")
    
    # Initialize with sample frames for optimization
    sample_frames = frames[:min(50, len(frames) // 3)]
    print(f"[INFO] Analyzing video characteristics and optimizing parameters...")
    opinfer.gater.analyze_and_optimize(sample_frames)
    opinfer.initialized = True
    opinfer.gater.engine.reset()
    
    # Process frames and collect data
    print(f"[INFO] Processing {len(frames)} frames with motion-gated inference...")
    text_queries = opinfer.gater.text_queries if opinfer.gater.text_queries else detection_queries
    
    start_time = time.time()
    for i, frame in enumerate(frames):
        output, stats = opinfer.gater.process_frame(frame)
        collect_detection_data(i + 1, stats, output, text_queries)
        
        if (i + 1) % 100 == 0:
            print(f"   [INFO] Processed {i + 1}/{len(frames)} frames...")
    
    processing_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_performance_metrics()
    
    print(f"\n[SUCCESS] Video processing complete!")
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS: {video_name}")
    print(f"{'='*60}")
    print(f"Total Frames Processed: {metrics['total_frames']}")
    print(f"Model Calls: {metrics['total_model_calls']}")
    print(f"Skip Rate: {metrics['skip_rate_pct']:.1f}%")
    print(f"Performance Improvement: {(metrics['skip_rate_pct']/100)*100:.1f}% fewer model calls")
    print(f"Average Motion Score: {metrics['avg_motion_score']:.2f}")
    print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    print(f"Total Detections: {metrics['total_detections']}")
    print(f"Average Detection Score: {metrics['avg_detection_score']:.3f}")
    print(f"Object Types Detected: {metrics['object_types_detected']}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Effective FPS: {metrics['total_frames']/processing_time:.2f}")
    print(f"{'='*60}\n")
    
    # Generate graphs
    print(f"[INFO] Generating performance graphs...")
    generate_graphs(video_name)
    
    return metrics


def main():
    """Main function to process videos and generate graphs."""
    print("\n" + "="*80)
    print("OPINFER VIDEO PERFORMANCE TEST")
    print("="*80)
    print("\nThis script tests Opinfer's motion-gated inference on test videos")
    print("and generates comprehensive performance visualizations.\n")
    
    video_dir = Path("vdo")
    if not video_dir.exists():
        print(f"[WARNING] Video directory 'vdo/' not found!")
        print(f"Please create the 'vdo/' folder and add your test videos.")
        return
    
    # Default test videos (users can modify this list)
    videos = {
        "slowtraffic": "vdo/slowtraffic.mp4",
        "fastbikedrive": "vdo/fastbikedrive.mp4",
        "fastcarnightdrivedashcam": "vdo/fastcarnightdrivedashcam.mp4",
    }
    
    # Check which videos exist
    available_videos = {}
    for video_name, video_path in videos.items():
        if Path(video_path).exists():
            available_videos[video_name] = video_path
        else:
            print(f"[WARNING] Video not found: {video_path}")
    
    if not available_videos:
        print(f"[ERROR] No test videos found in 'vdo/' directory!")
        print(f"Please add test videos to run the performance test.")
        return
    
    print(f"[INFO] Found {len(available_videos)} test video(s)")
    
    all_metrics = {}
    for video_name, video_path in available_videos.items():
        print(f"\n{'='*80}")
        print(f"Processing: {video_name}")
        print(f"{'='*80}")
        try:
            metrics = process_video_with_data_collection(video_path, video_name, max_frames=500)
            all_metrics[video_name] = metrics
        except Exception as e:
            print(f"[ERROR] Failed to process {video_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    if all_metrics:
        avg_skip_rate = np.mean([m['skip_rate_pct'] for m in all_metrics.values()])
        print(f"\nAverage Skip Rate Across All Videos: {avg_skip_rate:.1f}%")
        print(f"Average Performance Improvement: {avg_skip_rate:.1f}% fewer model calls")
        print(f"\nAll graphs saved to 'graphs/' folder")
        print(f"Check individual analysis graphs for detailed metrics!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


