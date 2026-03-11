"""
消融实验 - 验证各组件的贡献
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import sys

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.fbcsp import FilterBankCSPClassifier, FBCSPConfig
from config import mi_config as cfg


class AblationStudy:
    """Perform ablation study to validate component contributions."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """
        Args:
            X: Data array (n_trials, n_channels, n_samples)
            y: Labels array (n_trials,)
            n_splits: Number of CV folds
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.results: Dict[str, Dict] = {}
        
    def create_config_variants(self) -> Dict[str, FBCSPConfig]:
        """Create different configuration variants for ablation."""
        base_config = {
            'sample_rate': cfg.SAMPLE_RATE_HZ,
            'components_per_band': cfg.CSP_COMPONENTS_PER_BAND,
            'svm_kernel': cfg.SVM_KERNEL,
            'svm_c': cfg.SVM_C,
        }
        
        variants = {}
        
        # 1. Full model (your method)
        variants['Full_Model'] = FBCSPConfig(
            **base_config,
            filter_banks=cfg.FILTER_BANKS,
            use_riemann_tangent=True
        )
        
        # 2. Without Riemann (pure FBCSP)
        variants['No_Riemann'] = FBCSPConfig(
            **base_config,
            filter_banks=cfg.FILTER_BANKS,
            use_riemann_tangent=False
        )
        
        # 3. Without Filter Bank (single band)
        variants['No_FilterBank'] = FBCSPConfig(
            **base_config,
            filter_banks=[(8, 30)],  # Single wide band
            use_riemann_tangent=True
        )
        
        # 4. Without Tangent Space (Riemannian distance + kNN)
        # This would need a different implementation
        # For now, we compare with pure CSP
        
        # 5. Single band + CSP (classic)
        variants['Classic_CSP'] = FBCSPConfig(
            **base_config,
            filter_banks=[(8, 30)],
            use_riemann_tangent=False
        )
        
        # 6. Only mu rhythm
        variants['Mu_Only'] = FBCSPConfig(
            **base_config,
            filter_banks=[(8, 12)],
            use_riemann_tangent=True
        )
        
        # 7. Only beta rhythm
        variants['Beta_Only'] = FBCSPConfig(
            **base_config,
            filter_banks=[(16, 24)],
            use_riemann_tangent=True
        )
        
        # 8. Reduced CSP components
        # Create a copy of base_config and modify components_per_band
        reduced_config = base_config.copy()
        reduced_config['components_per_band'] = 3  # Reduced from 6
        variants['Reduced_Components'] = FBCSPConfig(
            **reduced_config,
            filter_banks=cfg.FILTER_BANKS,
            use_riemann_tangent=True
        )
        
        return variants
    
    def evaluate_config(self, config: FBCSPConfig, 
                        name: str) -> Dict:
        """Evaluate a single configuration with cross-validation."""
        print(f"  Evaluating: {name}")
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_kappas = []
        fold_train_times = []
        fold_infer_times = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Train
            import time
            model = FilterBankCSPClassifier(config)
            
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            
            # Test
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t0
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            
            fold_accuracies.append(acc)
            fold_kappas.append(kappa)
            fold_train_times.append(train_time)
            fold_infer_times.append(infer_time)
        
        return {
            'accuracy_mean': np.mean(fold_accuracies),
            'accuracy_std': np.std(fold_accuracies),
            'kappa_mean': np.mean(fold_kappas),
            'kappa_std': np.std(fold_kappas),
            'train_time_mean': np.mean(fold_train_times),
            'infer_time_mean': np.mean(fold_infer_times),
            'fold_accuracies': fold_accuracies
        }
    
    def run_study(self) -> pd.DataFrame:
        """Run complete ablation study."""
        print("=" * 60)
        print("Ablation Study")
        print("=" * 60)
        print(f"Data shape: {self.X.shape}")
        print(f"Classes: {np.unique(self.y)}")
        print(f"CV folds: {self.n_splits}")
        
        variants = self.create_config_variants()
        
        results = []
        for name, config in variants.items():
            result = self.evaluate_config(config, name)
            result['config_name'] = name
            results.append(result)
            self.results[name] = result
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def print_results(self):
        """Print formatted results table."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)
        print(f"{'Configuration':<25} {'Accuracy':<15} {'Kappa':<15} {'Train(s)':<12}")
        print("-" * 80)
        
        for name, result in self.results.items():
            acc_str = f"{result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}"
            kappa_str = f"{result['kappa_mean']:.3f}±{result['kappa_std']:.3f}"
            train_str = f"{result['train_time_mean']:.3f}"
            print(f"{name:<25} {acc_str:<15} {kappa_str:<15} {train_str:<12}")
        
        print("-" * 80)
        
        # Calculate relative improvements
        if 'Classic_CSP' in self.results and 'Full_Model' in self.results:
            baseline_acc = self.results['Classic_CSP']['accuracy_mean']
            full_acc = self.results['Full_Model']['accuracy_mean']
            improvement = (full_acc - baseline_acc) / baseline_acc * 100
            print(f"\nRelative improvement over Classic CSP: {improvement:+.1f}%")
        
        if 'No_Riemann' in self.results and 'Full_Model' in self.results:
            no_riemann_acc = self.results['No_Riemann']['accuracy_mean']
            full_acc = self.results['Full_Model']['accuracy_mean']
            riemann_contribution = (full_acc - no_riemann_acc) / no_riemann_acc * 100
            print(f"Riemann contribution: {riemann_contribution:+.1f}%")
        
        if 'No_FilterBank' in self.results and 'Full_Model' in self.results:
            no_fb_acc = self.results['No_FilterBank']['accuracy_mean']
            full_acc = self.results['Full_Model']['accuracy_mean']
            fb_contribution = (full_acc - no_fb_acc) / no_fb_acc * 100
            print(f"Filter Bank contribution: {fb_contribution:+.1f}%")
    
    def save_results(self, output_path: str = None):
        """Save results to CSV."""
        from config.algorithms_config import get_timestamped_filename, get_results_path
        
        if output_path is None:
            output_path = get_results_path(get_timestamped_filename('ablation_results', 'csv'))
        else:
            # If output_path is provided, still add timestamp
            base_name = output_path.replace('.csv', '')
            output_path = get_results_path(get_timestamped_filename(base_name, 'csv'))
        
        df = pd.DataFrame([
            {
                'config_name': name,
                **{k: v for k, v in result.items() if k != 'fold_accuracies'}
            }
            for name, result in self.results.items()
        ])
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        # Also save detailed JSON
        json_path = output_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to {json_path}")
    
    def plot_results(self, save_path: str = None):
        """Plot ablation results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data
            configs = list(self.results.keys())
            accuracies = [self.results[c]['accuracy_mean'] for c in configs]
            errors = [self.results[c]['accuracy_std'] for c in configs]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar plot with error bars
            x_pos = np.arange(len(configs))
            bars = ax.bar(x_pos, accuracies, yerr=errors, 
                         capsize=5, alpha=0.7, color='steelblue')
            
            # Highlight full model
            if 'Full_Model' in configs:
                full_idx = configs.index('Full_Model')
                bars[full_idx].set_color('darkgreen')
                bars[full_idx].set_alpha(0.9)
            
            ax.set_xlabel('Configuration', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Ablation Study: Component Contributions', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")


def main():
    """Example usage with synthetic data."""
    from config.algorithms_config import get_timestamped_filename, get_results_path
    
    print("Generating synthetic data for demonstration...")
    np.random.seed(42)
    
    n_trials = 200
    n_channels = 8
    n_samples = 1000
    
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.random.randint(0, 4, n_trials)
    
    # Run study
    study = AblationStudy(X, y, n_splits=3)
    df = study.run_study()
    study.print_results()
    study.save_results()
    
    # Save plot with timestamp
    plot_path = get_results_path(get_timestamped_filename('ablation_plot', 'png'))
    study.plot_results(plot_path)


if __name__ == "__main__":
    main()
