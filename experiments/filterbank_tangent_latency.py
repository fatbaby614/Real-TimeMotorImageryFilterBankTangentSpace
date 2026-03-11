"""
Filter Bank Tangent Space实时延迟性能基准测试
测量不同配置的端到端延迟和吞吐量
"""
from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms_collection import FilterBankTangentSpace
from config.algorithms_config import RANDOM_STATE


class FilterBankTangentSpaceLatencyBenchmark:
    """Benchmark latency and throughput for Filter Bank Tangent Space algorithm."""
    
    def __init__(self, n_channels: int = 22, sample_rate: float = 250.0):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.results: Dict[str, Dict] = {}
        
    def generate_synthetic_data(self, n_trials: int, trial_duration: float) -> np.ndarray:
        """Generate synthetic EEG-like data."""
        n_samples = int(trial_duration * self.sample_rate)
        # Generate data with some structure (not pure noise)
        data = np.random.randn(n_trials, self.n_channels, n_samples).astype(np.float32)
        # Add some correlated structure to simulate brain rhythms
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
            model_copy = FilterBankTangentSpace(
                n_bands=model.n_bands,
                estimator=model.estimator,
                metric=model.metric,
                classifier=model.classifier_name,
                n_features=model.n_features,
                fs=model.fs
            )
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
    
    def benchmark_sliding_window(self, model, trial_duration: float = 2.5,
                                  window_sizes: List[float] = None,
                                  step_sizes: List[float] = None) -> Dict:
        """Benchmark sliding window processing."""
        if window_sizes is None:
            window_sizes = [0.5, 1.0, 1.5, 2.0, 2.5]
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
    
    def benchmark_memory_usage(self, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """Estimate memory usage during training and inference."""
        import tracemalloc
        import gc
        
        gc.collect()
        tracemalloc.start()
        
        # Measure training memory
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        model.fit(X, y)
        new_mem, new_peak_mem = tracemalloc.get_traced_memory()
        
        training_memory_mb = (new_mem - current_mem) / 1024 / 1024
        
        # Measure inference memory
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        _ = model.predict(X[:10])
        new_mem, new_peak_mem = tracemalloc.get_traced_memory()
        
        inference_memory_mb = (new_mem - current_mem) / 1024 / 1024
        
        tracemalloc.stop()
        
        return {
            'training_mb': training_memory_mb,
            'inference_mb': inference_memory_mb
        }
    
    def run_full_benchmark(self, n_train_trials: int = 100, 
                           trial_duration: float = 2.5) -> Dict:
        """Run complete benchmark suite."""
        print("=" * 60)
        print("Filter Bank Tangent Space Real-time Latency Benchmark")
        print("=" * 60)
        
        # Generate data
        print("\n1. Generating synthetic data...")
        X_train = self.generate_synthetic_data(n_train_trials, trial_duration)
        y_train = np.random.randint(0, 4, n_train_trials)
        
        # Test different configurations
        configs = [
            ('Baseline_SVM', 9, 'oas', 'riemann', 'svm', 100),
            ('Baseline_LDA', 9, 'oas', 'riemann', 'lda', 100),
            ('Baseline_RF', 9, 'oas', 'riemann', 'rf', 100),
            ('No_FeatureSelection', 9, 'oas', 'riemann', 'svm', None),
            ('Single_Band', 1, 'oas', 'riemann', 'svm', 100),
            ('Bands_3', 3, 'oas', 'riemann', 'svm', 100),
            ('Bands_5', 5, 'oas', 'riemann', 'svm', 100),
            ('Features_50', 9, 'oas', 'riemann', 'svm', 50),
            ('Features_200', 9, 'oas', 'riemann', 'svm', 200),
        ]
        
        all_results = {}
        
        for name, n_bands, estimator, metric, classifier, n_features in configs:
            print(f"\n2. Benchmarking {name}...")
            print(f"   Config: {n_bands} bands, {estimator} estimator, {metric} metric, {classifier} classifier, {n_features} features")
            
            config = {
                'n_bands': n_bands,
                'estimator': estimator,
                'metric': metric,
                'classifier': classifier,
                'n_features': n_features,
                'fs': self.sample_rate
            }
            
            model = FilterBankTangentSpace(**config)
            
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
            
            # Memory usage
            print("   - Measuring memory usage...")
            memory_results = self.benchmark_memory_usage(model, X_train, y_train)
            
            all_results[name] = {
                'config': config,
                'training': train_results,
                'inference': infer_results,
                'sliding_window': sliding_results,
                'memory': memory_results
            }
        
        self.results = all_results
        return all_results
    
    def print_results(self):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        for method, results in self.results.items():
            config = results['config']
            print(f"\n{method}:")
            print("-" * 40)
            print(f"  Config: {config['n_bands']} bands, {config['estimator']} estimator, {config['classifier']} classifier")
            if config['n_features']:
                print(f"  Features: {config['n_features']}")
            else:
                print(f"  Features: All (no selection)")
            
            # Training
            train = results['training']
            print(f"  Training Time: {train['mean']:.3f} ± {train['std']:.3f} s")
            
            # Inference
            infer = results['inference']
            print(f"  Inference Time: {infer['mean_ms']:.2f} ± {infer['std_ms']:.2f} ms")
            print(f"  Throughput: {infer['throughput_trials_per_sec']:.1f} trials/sec")
            
            # Sliding window (1.5s window, 0.2s step)
            print(f"  Sliding Window Latency (1.5s window, 0.2s step):")
            if 1.5 in results['sliding_window'] and 0.2 in results['sliding_window'][1.5]:
                sw = results['sliding_window'][1.5][0.2]
                print(f"    - Processing: {sw['time_per_window_ms']:.2f} ms")
                print(f"    - Total Latency: {sw['latency_ms']:.2f} ms")
            
            # Memory
            mem = results['memory']
            print(f"  Memory Usage:")
            print(f"    - Training: {mem['training_mb']:.2f} MB")
            print(f"    - Inference: {mem['inference_mb']:.2f} MB")
    
    def save_results(self, output_path: str = None):
        """Save results to JSON."""
        from config.algorithms_config import get_timestamped_filename, get_results_path
        
        if output_path is None:
            output_path = get_results_path(get_timestamped_filename('fbts_latency_benchmark', 'json'))
        else:
            # If output_path is provided, still add timestamp
            base_name = output_path.replace('.json', '')
            output_path = get_results_path(get_timestamped_filename(base_name, 'json'))
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    def plot_results(self, save_path: str = None):
        """Plot benchmark results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data
            configs = list(self.results.keys())
            train_times = [self.results[c]['training']['mean'] for c in configs]
            infer_times = [self.results[c]['inference']['mean_ms'] for c in configs]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Training time plot
            x_pos = np.arange(len(configs))
            bars1 = ax1.bar(x_pos, train_times, alpha=0.7, color='steelblue')
            ax1.set_xlabel('Configuration', fontsize=12)
            ax1.set_ylabel('Training Time (s)', fontsize=12)
            ax1.set_title('Training Time Comparison', fontsize=14)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(configs, rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # Inference time plot
            bars2 = ax2.bar(x_pos, infer_times, alpha=0.7, color='coral')
            ax2.set_xlabel('Configuration', fontsize=12)
            ax2.set_ylabel('Inference Time (ms)', fontsize=12)
            ax2.set_title('Inference Time Comparison', fontsize=14)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(configs, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")


def main():
    """Run benchmark."""
    benchmark = FilterBankTangentSpaceLatencyBenchmark()
    results = benchmark.run_full_benchmark(n_train_trials=100)
    benchmark.print_results()
    benchmark.save_results()
    
    # Save plot with timestamp
    from config.algorithms_config import get_results_path, get_timestamped_filename
    plot_path = get_results_path(get_timestamped_filename('fbts_latency_plot', 'png'))
    benchmark.plot_results(plot_path)


if __name__ == "__main__":
    main()