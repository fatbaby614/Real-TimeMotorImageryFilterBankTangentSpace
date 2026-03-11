"""
实时延迟性能基准测试
测量不同算法的端到端延迟和吞吐量
"""
from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.fbcsp import FilterBankCSPClassifier, FBCSPConfig
from config import mi_config as cfg


class LatencyBenchmark:
    """Benchmark latency and throughput for different algorithms."""
    
    def __init__(self, n_channels: int = 8, sample_rate: float = 250.0):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.results: Dict[str, Dict] = {}
        
    def generate_synthetic_data(self, n_trials: int, trial_duration: float) -> np.ndarray:
        """Generate synthetic EEG-like data."""
        n_samples = int(trial_duration * self.sample_rate)
        # Generate data with some structure (not pure noise)
        data = np.random.randn(n_trials, self.n_channels, n_samples).astype(np.float32)
        # Add some correlated structure
        for i in range(n_trials):
            for ch in range(self.n_channels):
                # Add sine waves to simulate brain rhythms
                t = np.arange(n_samples) / self.sample_rate
                data[i, ch, :] += 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
                data[i, ch, :] += 0.3 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
        return data
    
    def benchmark_training_time(self, model, X: np.ndarray, y: np.ndarray, 
                                 n_runs: int = 5) -> Dict:
        """Measure training time."""
        times = []
        for _ in range(n_runs):
            model_copy = type(model)(model.config) if hasattr(model, 'config') else model
            start = time.perf_counter()
            model_copy.fit(X, y)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'trials': X.shape[0]
        }
    
    def benchmark_inference_time(self, model, X: np.ndarray, 
                                  n_runs: int = 100) -> Dict:
        """Measure single-trial inference time."""
        times = []
        # Warm up
        for _ in range(10):
            _ = model.predict(X[:1])
        
        # Benchmark
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X[:1])
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput_trials_per_sec': 1000.0 / np.mean(times)
        }
    
    def benchmark_sliding_window(self, model, trial_duration: float = 4.0,
                                  window_sizes: List[float] = None,
                                  step_sizes: List[float] = None) -> Dict:
        """Benchmark sliding window processing."""
        if window_sizes is None:
            window_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        if step_sizes is None:
            step_sizes = [0.1, 0.2, 0.3, 0.5]
        
        results = {}
        n_samples_total = int(trial_duration * self.sample_rate)
        
        for window_sec in window_sizes:
            window_samples = int(window_sec * self.sample_rate)
            if window_samples > n_samples_total:
                continue
                
            results[window_sec] = {}
            
            for step_sec in step_sizes:
                step_samples = int(step_sec * self.sample_rate)
                
                # Simulate sliding window
                n_windows = (n_samples_total - window_samples) // step_samples + 1
                
                # Generate test data
                X_window = np.random.randn(1, self.n_channels, window_samples).astype(np.float32)
                
                # Measure time for all windows
                start = time.perf_counter()
                for _ in range(n_windows):
                    _ = model.predict(X_window)
                elapsed = time.perf_counter() - start
                
                results[window_sec][step_sec] = {
                    'n_windows': n_windows,
                    'total_time_ms': elapsed * 1000,
                    'time_per_window_ms': elapsed * 1000 / n_windows,
                    'latency_ms': elapsed * 1000 / n_windows + step_sec * 1000
                }
        
        return results
    
    def run_full_benchmark(self, n_train_trials: int = 100, 
                           trial_duration: float = 4.0) -> Dict:
        """Run complete benchmark suite."""
        print("=" * 60)
        print("Real-time Latency Benchmark")
        print("=" * 60)
        
        # Generate data
        print("\n1. Generating synthetic data...")
        X_train = self.generate_synthetic_data(n_train_trials, trial_duration)
        y_train = np.random.randint(0, 4, n_train_trials)
        
        # Test different configurations
        configs = [
            ('CSP', False),
            ('Riemann', True),
        ]
        
        all_results = {}
        
        for name, use_riemann in configs:
            print(f"\n2. Benchmarking {name}...")
            
            config = FBCSPConfig(
                sample_rate=self.sample_rate,
                filter_banks=cfg.FILTER_BANKS,
                components_per_band=cfg.CSP_COMPONENTS_PER_BAND,
                svm_kernel=cfg.SVM_KERNEL,
                svm_c=cfg.SVM_C,
                use_riemann_tangent=use_riemann
            )
            
            model = FilterBankCSPClassifier(config)
            
            # Training time
            print("   - Measuring training time...")
            train_results = self.benchmark_training_time(model, X_train, y_train)
            
            # Fit model for inference tests
            model.fit(X_train, y_train)
            
            # Inference time
            print("   - Measuring inference time...")
            infer_results = self.benchmark_inference_time(model, X_train)
            
            # Sliding window
            print("   - Measuring sliding window performance...")
            sliding_results = self.benchmark_sliding_window(model, trial_duration)
            
            all_results[name] = {
                'training': train_results,
                'inference': infer_results,
                'sliding_window': sliding_results
            }
        
        self.results = all_results
        return all_results
    
    def print_results(self):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        for method, results in self.results.items():
            print(f"\n{method}:")
            print("-" * 40)
            
            # Training
            train = results['training']
            print(f"  Training Time: {train['mean']:.3f} ± {train['std']:.3f} s")
            
            # Inference
            infer = results['inference']
            print(f"  Inference Time: {infer['mean_ms']:.2f} ± {infer['std_ms']:.2f} ms")
            print(f"  Throughput: {infer['throughput_trials_per_sec']:.1f} trials/sec")
            
            # Sliding window
            print(f"  Sliding Window Latency (1.5s window, 0.2s step):")
            if 1.5 in results['sliding_window'] and 0.2 in results['sliding_window'][1.5]:
                sw = results['sliding_window'][1.5][0.2]
                print(f"    - Processing: {sw['time_per_window_ms']:.2f} ms")
                print(f"    - Total Latency: {sw['latency_ms']:.2f} ms")
    
    def save_results(self, output_path: str = None):
        """Save results to JSON."""
        from config.algorithms_config import get_timestamped_filename, get_results_path
        
        if output_path is None:
            output_path = get_results_path(get_timestamped_filename('latency_benchmark', 'json'))
        else:
            # If output_path is provided, still add timestamp
            base_name = output_path.replace('.json', '')
            output_path = get_results_path(get_timestamped_filename(base_name, 'json'))
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    """Run benchmark."""
    benchmark = LatencyBenchmark()
    results = benchmark.run_full_benchmark(n_train_trials=100)
    benchmark.print_results()
    benchmark.save_results()


if __name__ == "__main__":
    main()
