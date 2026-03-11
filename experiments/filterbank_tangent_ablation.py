"""
Filter Bank Tangent Space消融实验
验证各组件对性能的贡献
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
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from algorithms_collection import FilterBankTangentSpace
from config.algorithms_config import RANDOM_STATE, N_SPLITS


class FilterBankTangentSpaceAblationStudy:
    """Perform ablation study for Filter Bank Tangent Space algorithm."""
    
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
        
    def create_config_variants(self) -> Dict[str, Dict]:
        """Create different configuration variants for ablation."""
        variants = {}
        
        # 1. Full model (baseline)
        variants['Full_Model'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        # 2. Without feature selection (use all features)
        variants['No_FeatureSelection'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': None,  # No feature selection
            'fs': 250
        }
        
        # 3. Single band (no filter bank)
        variants['Single_Band'] = {
            'n_bands': 1,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        # 4. Different number of bands
        variants['Bands_3'] = {
            'n_bands': 3,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        variants['Bands_5'] = {
            'n_bands': 5,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        # 5. Different estimators
        variants['Estimator_LWF'] = {
            'n_bands': 9,
            'estimator': 'lwf',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        variants['Estimator_SCM'] = {
            'n_bands': 9,
            'estimator': 'scm',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        # 6. Different metrics
        variants['Metric_Euclid'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'euclid',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        variants['Metric_LogEuclid'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'logeuclid',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        # 7. Different classifiers
        variants['Classifier_LDA'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'lda',
            'n_features': 100,
            'fs': 250
        }
        
        variants['Classifier_RF'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'rf',
            'n_features': 100,
            'fs': 250
        }
        
        # 8. Different feature numbers
        variants['Features_50'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 50,
            'fs': 250
        }
        
        variants['Features_200'] = {
            'n_bands': 9,
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 200,
            'fs': 250
        }
        
        # 9. Only specific frequency bands
        variants['Mu_Only'] = {
            'n_bands': 2,  # 4-8Hz, 8-12Hz
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        variants['Beta_Only'] = {
            'n_bands': 3,  # 12-16Hz, 16-20Hz, 20-24Hz
            'estimator': 'oas',
            'metric': 'riemann',
            'classifier': 'svm',
            'n_features': 100,
            'fs': 250
        }
        
        return variants
    
    def evaluate_config(self, config: Dict, name: str) -> Dict:
        """Evaluate a single configuration with cross-validation."""
        print(f"  Evaluating: {name}")
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_STATE)
        
        fold_accuracies = []
        fold_kappas = []
        fold_train_times = []
        fold_infer_times = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            # Create model
            import time
            model = FilterBankTangentSpace(**config)
            
            # Train
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
        print("Filter Bank Tangent Space Ablation Study")
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
        if 'Single_Band' in self.results and 'Full_Model' in self.results:
            single_acc = self.results['Single_Band']['accuracy_mean']
            full_acc = self.results['Full_Model']['accuracy_mean']
            fb_contribution = (full_acc - single_acc) / single_acc * 100
            print(f"\nFilter Bank contribution: {fb_contribution:+.1f}%")
        
        if 'No_FeatureSelection' in self.results and 'Full_Model' in self.results:
            no_fs_acc = self.results['No_FeatureSelection']['accuracy_mean']
            full_acc = self.results['Full_Model']['accuracy_mean']
            fs_contribution = (full_acc - no_fs_acc) / no_fs_acc * 100
            print(f"Feature selection contribution: {fs_contribution:+.1f}%")
        
        # Find best configuration
        best_config = max(self.results.items(), key=lambda x: x[1]['accuracy_mean'])
        print(f"\nBest configuration: {best_config[0]} (Accuracy: {best_config[1]['accuracy_mean']:.3f})")
    
    def save_results(self, output_path: str = None):
        """Save results to CSV."""
        from config.algorithms_config import get_timestamped_filename, get_results_path
        
        if output_path is None:
            output_path = get_results_path(get_timestamped_filename('fbts_ablation_results', 'csv'))
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
            fig, ax = plt.subplots(figsize=(16, 8))
            
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
            ax.set_title('Filter Bank Tangent Space Ablation Study', fontsize=14)
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
    """Main function with argument parsing."""
    from config.algorithms_config import get_timestamped_filename, get_results_path
    import data_loader_moabb as data_loader
    
    parser = argparse.ArgumentParser(description="Filter Bank Tangent Space ablation study")
    parser.add_argument("--subjects", type=str, default="1~9",
                       help="Subject range (e.g., '1~9', '1,3,5~7')")
    parser.add_argument("--dataset", type=str, default="BCI_IV_2A",
                       help="Dataset name (BCI_IV_2A, PhysionetMI, Schirrmeister2017)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", default=None, 
                       help="Output CSV file base name (timestamp will be added)")
    args = parser.parse_args()
    
    # Parse subjects
    if args.subjects == "all":
        if args.dataset == "BCI_IV_2A":
            subjects = list(range(1, 10))
        elif args.dataset == "PhysionetMI":
            subjects = list(range(1, 110))
        else:
            subjects = list(range(1, 15))
    elif "~" in args.subjects:
        start, end = map(int, args.subjects.split("~"))
        subjects = list(range(start, end + 1))
    else:
        subjects = [int(s) for s in args.subjects.split(",")]
    
    print(f"Running ablation study on {args.dataset} for subjects: {subjects}")
    
    # Load data for first subject
    X, y, _ = data_loader.load_single_subject_moabb(subjects[0], use_test_data=False, dataset=args.dataset)
    
    # Run study
    study = FilterBankTangentSpaceAblationStudy(X, y, n_splits=args.n_splits)
    df = study.run_study()
    study.print_results()
    
    # Save results with timestamp
    if args.output:
        study.save_results(args.output)
        # Save plot with timestamp
        plot_path = get_results_path(get_timestamped_filename(args.output.replace('.csv', ''), 'png'))
    else:
        study.save_results()
        # Save plot with timestamp
        plot_path = get_results_path(get_timestamped_filename('fbts_ablation_plot', 'png'))
    
    study.plot_results(plot_path)


if __name__ == "__main__":
    main()