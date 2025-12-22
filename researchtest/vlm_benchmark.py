"""
Comprehensive VLM (Vision-Language Model) Benchmarking with Opinfer
Tests multiple VLM models and compares performance metrics
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
import torch
from dataclasses import dataclass, asdict

# Optional imports with fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available, graphs will be skipped")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available, using basic data structures")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opinfer import OptimizedInference, load_video_frames
from opinfer.models import ModelLoader
from opinfer.adaptive import AdaptiveMotionGater
from opinfer.core import MotionGatedInference
from opinfer.detectors import VideoCharacteristicDetector


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""
    model_name: str
    model_type: str
    technique: str
    video_name: str
    
    # Performance metrics
    total_frames: int
    model_calls: int
    skipped_frames: int
    skip_rate_pct: float
    avg_inference_ms: float
    total_time_seconds: float
    effective_fps: float
    
    # Optimization metrics
    motion_threshold: float
    min_frames_between_calls: int
    
    # Video characteristics
    video_motion_pattern: Optional[str] = None
    video_lighting: Optional[str] = None


class VLMResearchBenchmark:
    """Comprehensive benchmarking suite for VLM models."""
    
    # Available VLM models (detectors/vision-language models)
    VLM_MODELS = [
        ("owlvit-base", "detector", "google/owlvit-base-patch16"),
        ("owlvit-large", "detector", "google/owlvit-large-patch14"),
    ]
    
    # Available classifier models (ViT-based)
    CLASSIFIER_MODELS = [
        ("vit_base_patch16_224", "classifier"),
        ("vit_large_patch16_224", "classifier"),
        ("deit_base_patch16_224", "classifier"),
    ]
    
    # Test videos
    TEST_VIDEOS = [
        ("trafficcam30fps.mp4", "static_traffic"),
        ("mediumpace30fps.mp4", "moderate_motion"),
        ("mediumspeeddrone30fps.mp4", "fast_motion"),
    ]
    
    # Text queries for detector models
    DETECTOR_QUERIES = ["person", "car", "road", "building", "tree", "traffic light", "sky", "vehicle"]
    
    def __init__(
        self, 
        video_dir: str = "../vdo", 
        max_frames: int = 300,
        fast_mode: bool = True,
        skip_optimization: bool = False,
        ultra_fast: bool = False,
    ):
        """
        Initialize benchmark suite.
        
        Args:
            video_dir: Directory containing test videos
            max_frames: Maximum frames to process per video
            fast_mode: If True, uses faster optimization settings
            skip_optimization: If True, uses recommended parameters without optimization
            ultra_fast: If True, uses minimal optimization (3 iterations, 10 frames) for speed
        """
        self.video_dir = Path(video_dir)
        self.max_frames = max_frames
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fast_mode = fast_mode
        self.skip_optimization = skip_optimization
        self.ultra_fast = ultra_fast
        self.results: List[BenchmarkResult] = []
        
        print(f"🚀 VLM Research Benchmark Initialized")
        print(f"   Device: {self.device}")
        print(f"   Max frames per video: {max_frames}")
        print(f"   Fast mode: {fast_mode}")
        print(f"   Skip optimization: {skip_optimization}")
        print(f"   Ultra fast mode: {ultra_fast}")
        print(f"   VLM models to test: {len(self.VLM_MODELS)}")
        print(f"   Test videos: {len(self.TEST_VIDEOS)}")
    
    def load_video(self, video_name: str) -> List[np.ndarray]:
        """Load frames from a video file."""
        video_path = self.video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\n📹 Loading video: {video_name}")
        frames = load_video_frames(str(video_path), max_frames=self.max_frames)
        print(f"   Loaded {len(frames)} frames")
        return frames
    
    def benchmark_model(
        self,
        model_name: str,
        model_type: str,
        model_path: Optional[str] = None,
        technique: str = "motion_gating",
        video_frames: List[np.ndarray] = None,
        video_name: str = "unknown",
    ) -> BenchmarkResult:
        """
        Benchmark a single model on a video.
        
        Args:
            model_name: Name of the model
            model_type: "classifier" or "detector"
            model_path: Optional model path for custom models
            technique: "motion_gating" or "queuing"
            video_frames: List of frames to process
            video_name: Name of the video being tested
            
        Returns:
            BenchmarkResult with all metrics
        """
        print(f"\n{'='*80}")
        print(f"🧪 Testing: {model_name} ({model_type}) on {video_name}")
        print(f"   Technique: {technique}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Load model
            if model_type == "detector":
                model, processor = ModelLoader.load_detector(model_name, device=self.device)
                text_queries = self.DETECTOR_QUERIES
            else:
                model = ModelLoader.load_classifier(model_name, device=self.device)
                processor = None
                text_queries = None
            
            # Create optimized inference
            opt_inf = OptimizedInference(
                model_name=model_name,
                model_type=model_type,
                device=self.device,
                technique=technique,
                auto_optimize=not self.skip_optimization,
                target_skip_rate=40.0,
            )
            
            # Ultra-fast mode: skip optimization, use recommended params only
            if self.ultra_fast and not self.skip_optimization:
                # Use minimal frames for analysis only
                sample_frames = video_frames[:min(10, len(video_frames))]
                
                # Do analysis only (no iterative optimization)
                print("\n📊 Analyzing video (ultra-fast: skipping optimization)...")
                opt_inf.gater.detector = VideoCharacteristicDetector()
                opt_inf.gater.video_chars = opt_inf.gater.detector.analyze_video(sample_frames)
                
                # Use recommended parameters directly (no optimization loop)
                opt_inf.gater.motion_threshold = float(np.mean(opt_inf.gater.video_chars.recommended_threshold_range))
                opt_inf.gater.min_frames_between_calls = opt_inf.gater.video_chars.recommended_min_frames
                opt_inf.gater.optimization_result = None
                
                # Create engine with recommended params
                opt_inf.gater.engine = MotionGatedInference(
                    model=opt_inf.gater.model,
                    device=opt_inf.gater.device,
                    motion_threshold=opt_inf.gater.motion_threshold,
                    min_frames_between_calls=opt_inf.gater.min_frames_between_calls,
                )
                
                # Store processor/queries for detector models
                opt_inf.gater.engine.processor = opt_inf.gater.processor
                opt_inf.gater.engine.text_queries = opt_inf.gater.text_queries
                
                print(f"✅ Using recommended parameters (no optimization):")
                print(f"   Motion threshold: {opt_inf.gater.motion_threshold:.2f}")
                print(f"   Min frames: {opt_inf.gater.min_frames_between_calls}")
                
            elif self.fast_mode:
                sample_frames = video_frames[:min(20, len(video_frames))]
                opt_inf.gater.optimization_sample_frames = 20
                opt_inf.gater.analyze_and_optimize(sample_frames)
            else:
                sample_frames = video_frames[:min(50, len(video_frames))]
                opt_inf.gater.analyze_and_optimize(sample_frames)
            
            opt_inf.initialized = True
            
            # Get video characteristics
            video_chars = opt_inf.gater.video_chars
            
            # Process all frames
            for i, frame in enumerate(video_frames):
                opt_inf.process_frame(frame)
            
            # Get final statistics
            if technique == "motion_gating":
                stats = opt_inf.gater.get_current_stats()
            else:
                stats = opt_inf.queuer.get_stats()
            
            total_time = time.time() - start_time
            
            # Extract parameters
            if technique == "motion_gating":
                motion_threshold = opt_inf.gater.motion_threshold
                min_frames = opt_inf.gater.min_frames_between_calls
            else:
                motion_threshold = 0.0
                min_frames = 0
            
            result = BenchmarkResult(
                model_name=model_name,
                model_type=model_type,
                technique=technique,
                video_name=video_name,
                total_frames=stats.get("total_frames", len(video_frames)),
                model_calls=stats.get("model_calls", 0),
                skipped_frames=stats.get("skipped_frames", 0),
                skip_rate_pct=stats.get("skip_rate_pct", 0.0),
                avg_inference_ms=stats.get("avg_inference_ms", 0.0),
                total_time_seconds=total_time,
                effective_fps=stats.get("effective_fps", 0.0),
                motion_threshold=motion_threshold,
                min_frames_between_calls=min_frames,
                video_motion_pattern=video_chars.motion_pattern if video_chars else None,
                video_lighting=video_chars.lighting_condition if video_chars else None,
            )
            
            print(f"\n✅ Completed: {model_name}")
            print(f"   Skip rate: {result.skip_rate_pct:.1f}%")
            print(f"   Effective FPS: {result.effective_fps:.2f}")
            print(f"   Total time: {result.total_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty result on error
            return BenchmarkResult(
                model_name=model_name,
                model_type=model_type,
                technique=technique,
                video_name=video_name,
                total_frames=len(video_frames),
                model_calls=0,
                skipped_frames=0,
                skip_rate_pct=0.0,
                avg_inference_ms=0.0,
                total_time_seconds=time.time() - start_time,
                effective_fps=0.0,
                motion_threshold=0.0,
                min_frames_between_calls=0,
            )
    
    def run_all_benchmarks(self, techniques: List[str] = ["motion_gating"]):
        """Run benchmarks for all models and videos."""
        print(f"\n{'='*80}")
        print(f"🎯 STARTING COMPREHENSIVE VLM BENCHMARK")
        print(f"{'='*80}")
        
        # Test VLM models (detectors)
        for model_name, model_type, _ in self.VLM_MODELS:
            for video_file, video_scenario in self.TEST_VIDEOS:
                try:
                    frames = self.load_video(video_file)
                    
                    for technique in techniques:
                        result = self.benchmark_model(
                            model_name=model_name,
                            model_type=model_type,
                            technique=technique,
                            video_frames=frames,
                            video_name=video_scenario,
                        )
                        self.results.append(result)
                except Exception as e:
                    print(f"❌ Failed to test {model_name} on {video_file}: {e}")
        
        # Test classifier models for comparison
        for model_name, model_type in self.CLASSIFIER_MODELS:
            for video_file, video_scenario in self.TEST_VIDEOS:
                try:
                    frames = self.load_video(video_file)
                    
                    for technique in techniques:
                        result = self.benchmark_model(
                            model_name=model_name,
                            model_type=model_type,
                            technique=technique,
                            video_frames=frames,
                            video_name=video_scenario,
                        )
                        self.results.append(result)
                except Exception as e:
                    print(f"❌ Failed to test {model_name} on {video_file}: {e}")
    
    def save_results(self, output_file: str = "vlm_benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / output_file
        results_dict = [asdict(r) for r in self.results]
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def generate_dataframe(self):
        """Convert results to pandas DataFrame or dict."""
        data = [asdict(r) for r in self.results]
        if PANDAS_AVAILABLE:
            return pd.DataFrame(data)
        return data
    
    def generate_graphs(self, output_dir: str = "graphs"):
        """Generate comprehensive graphs comparing models."""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib not available, skipping graph generation")
            return
        
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        df = self.generate_dataframe()
        
        if PANDAS_AVAILABLE:
            if df.empty:
                print("⚠️  No results to graph")
                return
        else:
            if not df or len(df) == 0:
                print("⚠️  No results to graph")
                return
        
        print(f"\n📊 Generating graphs in: {output_path}")
        
        if not PANDAS_AVAILABLE:
            print("⚠️  pandas required for graph generation")
            return
        
        # 1. Skip Rate Comparison by Model
        fig, ax = plt.subplots(figsize=(12, 6))
        model_skip_rates = df.groupby('model_name')['skip_rate_pct'].mean().sort_values(ascending=False)
        model_skip_rates.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Average Skip Rate by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Skip Rate (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'skip_rate_by_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Effective FPS Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        model_fps = df.groupby('model_name')['effective_fps'].mean().sort_values(ascending=False)
        model_fps.plot(kind='bar', ax=ax, color='green')
        ax.set_title('Average Effective FPS by Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Effective FPS', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'fps_by_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance by Video Scenario
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Skip rate by scenario
        scenario_skip = df.groupby('video_name')['skip_rate_pct'].mean()
        scenario_skip.plot(kind='bar', ax=axes[0], color='coral')
        axes[0].set_title('Skip Rate by Video Scenario', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Video Scenario', fontsize=12)
        axes[0].set_ylabel('Skip Rate (%)', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # FPS by scenario
        scenario_fps = df.groupby('video_name')['effective_fps'].mean()
        scenario_fps.plot(kind='bar', ax=axes[1], color='mediumseagreen')
        axes[1].set_title('Effective FPS by Video Scenario', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Video Scenario', fontsize=12)
        axes[1].set_ylabel('Effective FPS', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_by_scenario.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Model Type Comparison (VLM vs Classifier)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        type_skip = df.groupby('model_type')['skip_rate_pct'].mean()
        type_skip.plot(kind='bar', ax=axes[0], color=['purple', 'orange'])
        axes[0].set_title('Skip Rate: VLM (Detector) vs Classifier', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Model Type', fontsize=12)
        axes[0].set_ylabel('Skip Rate (%)', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        type_fps = df.groupby('model_type')['effective_fps'].mean()
        type_fps.plot(kind='bar', ax=axes[1], color=['purple', 'orange'])
        axes[1].set_title('Effective FPS: VLM (Detector) vs Classifier', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Model Type', fontsize=12)
        axes[1].set_ylabel('Effective FPS', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'vlm_vs_classifier.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Comprehensive Comparison Heatmap
        pivot_skip = df.pivot_table(
            values='skip_rate_pct',
            index='model_name',
            columns='video_name',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_skip.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot_skip.columns)))
        ax.set_yticks(range(len(pivot_skip.index)))
        ax.set_xticklabels(pivot_skip.columns)
        ax.set_yticklabels(pivot_skip.index)
        ax.set_title('Skip Rate Heatmap: Model × Video Scenario', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_skip.index)):
            for j in range(len(pivot_skip.columns)):
                text = ax.text(j, i, f'{pivot_skip.iloc[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Skip Rate (%)')
        plt.tight_layout()
        plt.savefig(output_path / 'skip_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Detailed Performance Metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average inference time
        inf_time = df.groupby('model_name')['avg_inference_ms'].mean().sort_values()
        inf_time.plot(kind='barh', ax=axes[0, 0], color='red')
        axes[0, 0].set_title('Average Inference Time by Model', fontweight='bold')
        axes[0, 0].set_xlabel('Inference Time (ms)')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Total processing time
        total_time = df.groupby('model_name')['total_time_seconds'].mean().sort_values()
        total_time.plot(kind='barh', ax=axes[0, 1], color='blue')
        axes[0, 1].set_title('Total Processing Time by Model', fontweight='bold')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Model calls efficiency
        calls_efficiency = df.groupby('model_name').agg({
            'total_frames': 'mean',
            'model_calls': 'mean'
        })
        calls_efficiency['efficiency'] = (calls_efficiency['model_calls'] / calls_efficiency['total_frames']) * 100
        calls_efficiency['efficiency'].sort_values().plot(kind='barh', ax=axes[1, 0], color='green')
        axes[1, 0].set_title('Model Call Efficiency (% of frames)', fontweight='bold')
        axes[1, 0].set_xlabel('Efficiency (%)')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Skip rate distribution
        df['skip_rate_pct'].hist(bins=20, ax=axes[1, 1], color='teal', edgecolor='black')
        axes[1, 1].set_title('Skip Rate Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Skip Rate (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'detailed_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated {6} graph files")
    
    def generate_report(self, output_file: str = "benchmark_report.md"):
        """Generate a markdown report with results."""
        output_path = Path(__file__).parent / output_file
        df = self.generate_dataframe()
        
        with open(output_path, 'w') as f:
            f.write("# VLM Benchmark Results Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"- Total tests: {len(self.results)}\n")
            
            if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
                f.write(f"- Models tested: {df['model_name'].nunique()}\n")
                f.write(f"- Videos tested: {df['video_name'].nunique()}\n")
                f.write(f"- Average skip rate: {df['skip_rate_pct'].mean():.2f}%\n")
                f.write(f"- Average effective FPS: {df['effective_fps'].mean():.2f}\n\n")
                
                f.write("## Model Performance Summary\n\n")
                summary = df.groupby('model_name').agg({
                    'skip_rate_pct': 'mean',
                    'effective_fps': 'mean',
                    'avg_inference_ms': 'mean',
                    'total_time_seconds': 'mean',
                }).round(2)
                summary.columns = ['Avg Skip Rate (%)', 'Avg FPS', 'Avg Inference (ms)', 'Avg Total Time (s)']
                
                # Try to_markdown, fallback to simple table if tabulate missing
                try:
                    f.write(summary.to_markdown())
                except ImportError:
                    # Fallback: simple markdown table
                    f.write("| Model | Avg Skip Rate (%) | Avg FPS | Avg Inference (ms) | Avg Total Time (s) |\n")
                    f.write("|-------|-------------------|---------|-------------------|-------------------|\n")
                    for model, row in summary.iterrows():
                        f.write(f"| {model} | {row['Avg Skip Rate (%)']:.2f} | {row['Avg FPS']:.2f} | {row['Avg Inference (ms)']:.2f} | {row['Avg Total Time (s)']:.2f} |\n")
                
                f.write("\n\n")
                
                f.write("## Performance by Video Scenario\n\n")
                scenario_summary = df.groupby('video_name').agg({
                    'skip_rate_pct': 'mean',
                    'effective_fps': 'mean',
                }).round(2)
                scenario_summary.columns = ['Avg Skip Rate (%)', 'Avg FPS']
                
                try:
                    f.write(scenario_summary.to_markdown())
                except ImportError:
                    f.write("| Video Scenario | Avg Skip Rate (%) | Avg FPS |\n")
                    f.write("|----------------|-------------------|----------|\n")
                    for scenario, row in scenario_summary.iterrows():
                        f.write(f"| {scenario} | {row['Avg Skip Rate (%)']:.2f} | {row['Avg FPS']:.2f} |\n")
                
                f.write("\n\n")
                
                f.write("## Detailed Results\n\n")
                try:
                    f.write(df.to_markdown(index=False))
                except ImportError:
                    # Fallback: CSV-like format
                    f.write("| Model | Type | Video | Skip Rate % | FPS | Inference (ms) | Total Time (s) |\n")
                    f.write("|-------|------|-------|-------------|-----|----------------|----------------|\n")
                    for _, row in df.iterrows():
                        f.write(f"| {row['model_name']} | {row['model_type']} | {row['video_name']} | "
                               f"{row['skip_rate_pct']:.2f} | {row['effective_fps']:.2f} | "
                               f"{row['avg_inference_ms']:.2f} | {row['total_time_seconds']:.2f} |\n")
            else:
                # Basic report without pandas
                unique_models = set(r.model_name for r in self.results)
                unique_videos = set(r.video_name for r in self.results)
                avg_skip = sum(r.skip_rate_pct for r in self.results) / len(self.results) if self.results else 0
                avg_fps = sum(r.effective_fps for r in self.results) / len(self.results) if self.results else 0
                
                f.write(f"- Models tested: {len(unique_models)}\n")
                f.write(f"- Videos tested: {len(unique_videos)}\n")
                f.write(f"- Average skip rate: {avg_skip:.2f}%\n")
                f.write(f"- Average effective FPS: {avg_fps:.2f}\n\n")
                
                f.write("## Detailed Results\n\n")
                f.write("| Model | Type | Video | Skip Rate % | FPS | Inference (ms) |\n")
                f.write("|-------|------|-------|-------------|-----|----------------|\n")
                for r in self.results:
                    f.write(f"| {r.model_name} | {r.model_type} | {r.video_name} | {r.skip_rate_pct:.2f} | {r.effective_fps:.2f} | {r.avg_inference_ms:.2f} |\n")
            
            f.write("\n")
        
        print(f"📄 Report saved to: {output_path}")


def main():
    """Run comprehensive VLM benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM Benchmark with Opinfer')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (fewer iterations)')
    parser.add_argument('--no-opt', action='store_true', help='Skip optimization, use recommended parameters only')
    parser.add_argument('--ultra-fast', action='store_true', help='Ultra-fast mode: minimal optimization (3-5 min total)')
    parser.add_argument('--frames', type=int, default=150, help='Max frames per video (default: 150)')
    args = parser.parse_args()
    
    # Ultra-fast overrides other settings
    if args.ultra_fast:
        args.fast = True
        args.no_opt = False  # Still do analysis, just skip iterative optimization
    
    benchmark = VLMResearchBenchmark(
        video_dir="../vdo",
        max_frames=args.frames,
        fast_mode=args.fast,
        skip_optimization=args.no_opt,
        ultra_fast=args.ultra_fast,
    )
    
    # Run all benchmarks
    benchmark.run_all_benchmarks(techniques=["motion_gating"])
    
    # Save results
    benchmark.save_results()
    
    # Generate graphs
    benchmark.generate_graphs()
    
    # Generate report
    benchmark.generate_report()
    
    print(f"\n{'='*80}")
    print(f"🎉 BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Results saved to JSON")
    print(f"✅ Graphs generated in 'graphs/' folder")
    print(f"✅ Report generated as markdown")


if __name__ == "__main__":
    main()

