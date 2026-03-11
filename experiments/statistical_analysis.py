"""
统计分析工具 - 用于论文中的显著性检验
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))


class StatisticalAnalyzer:
    """Perform statistical analysis for algorithm comparison."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        
    def paired_ttest(self, scores_a: np.ndarray, scores_b: np.ndarray,
                     method_a: str = "Method A", method_b: str = "Method B") -> Dict:
        """
        Perform paired t-test between two methods.
        
        Returns:
            Dictionary with test statistics
        """
        # Check normality
        _, p_normal_a = stats.shapiro(scores_a)
        _, p_normal_b = stats.shapiro(scores_b)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        diff = scores_a - scores_b
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        # Confidence interval for mean difference
        n = len(diff)
        se = stats.sem(diff)
        ci = stats.t.interval(1 - self.alpha, n - 1, loc=np.mean(diff), scale=se)
        
        return {
            'method_a': method_a,
            'method_b': method_b,
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'std_a': np.std(scores_a, ddof=1),
            'std_b': np.std(scores_b, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'normality_a': p_normal_a > 0.05,
            'normality_b': p_normal_b > 0.05
        }
    
    def wilcoxon_signed_rank(self, scores_a: np.ndarray, scores_b: np.ndarray,
                             method_a: str = "Method A", method_b: str = "Method B") -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative).
        """
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        
        # Effect size (r = Z / sqrt(N))
        z_stat = stats.norm.ppf(1 - p_value / 2)
        effect_size_r = z_stat / np.sqrt(len(scores_a))
        
        return {
            'method_a': method_a,
            'method_b': method_b,
            'median_a': np.median(scores_a),
            'median_b': np.median(scores_b),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size_r': effect_size_r
        }
    
    def friedman_test(self, scores_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Perform Friedman test for multiple methods comparison.
        
        Args:
            scores_dict: Dictionary mapping method names to score arrays
        """
        methods = list(scores_dict.keys())
        scores_matrix = np.column_stack([scores_dict[m] for m in methods])
        
        statistic, p_value = stats.friedmanchisquare(*[scores_matrix[:, i] 
                                                        for i in range(len(methods))])
        
        return {
            'methods': methods,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_subjects': len(scores_matrix)
        }
    
    def bonferroni_correction(self, p_values: List[float], 
                              n_comparisons: Optional[int] = None) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons."""
        if n_comparisons is None:
            n_comparisons = len(p_values)
        
        corrected = [min(p * n_comparisons, 1.0) for p in p_values]
        return corrected
    
    def compare_all_pairs(self, scores_dict: Dict[str, np.ndarray],
                         correction: str = "bonferroni") -> pd.DataFrame:
        """
        Compare all pairs of methods with multiple comparison correction.
        
        Returns:
            DataFrame with comparison results
        """
        methods = list(scores_dict.keys())
        n_methods = len(methods)
        
        results = []
        raw_p_values = []
        comparisons = []
        
        # Collect all pairwise comparisons
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method_a, method_b = methods[i], methods[j]
                result = self.paired_ttest(
                    scores_dict[method_a], 
                    scores_dict[method_b],
                    method_a, method_b
                )
                results.append(result)
                raw_p_values.append(result['p_value'])
                comparisons.append(f"{method_a} vs {method_b}")
        
        # Apply correction
        if correction == "bonferroni":
            corrected_p = self.bonferroni_correction(raw_p_values)
        else:
            corrected_p = raw_p_values
        
        # Update results with corrected p-values
        for i, result in enumerate(results):
            result['p_value_corrected'] = corrected_p[i]
            result['significant_corrected'] = corrected_p[i] < self.alpha
        
        return pd.DataFrame(results)
    
    def generate_report(self, scores_dict: Dict[str, np.ndarray],
                       metrics: List[str] = None) -> str:
        """Generate a formatted statistical report."""
        if metrics is None:
            metrics = ['accuracy', 'kappa']
        
        report = []
        report.append("=" * 70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Descriptive statistics
        report.append("\n1. DESCRIPTIVE STATISTICS")
        report.append("-" * 70)
        report.append(f"{'Method':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'N':<6}")
        report.append("-" * 70)
        
        for method, scores in scores_dict.items():
            report.append(
                f"{method:<20} "
                f"{np.mean(scores):<12.4f} "
                f"{np.std(scores, ddof=1):<12.4f} "
                f"{np.median(scores):<12.4f} "
                f"{len(scores):<6}"
            )
        
        # Friedman test
        report.append("\n2. FRIEDMAN TEST (Overall Comparison)")
        report.append("-" * 70)
        friedman = self.friedman_test(scores_dict)
        report.append(f"Statistic: {friedman['statistic']:.4f}")
        report.append(f"p-value: {friedman['p_value']:.4f}")
        report.append(f"Significant: {'Yes' if friedman['significant'] else 'No'}")
        
        # Pairwise comparisons
        report.append("\n3. PAIRWISE COMPARISONS (Paired t-test)")
        report.append("-" * 70)
        
        df_comparisons = self.compare_all_pairs(scores_dict)
        
        report.append(
            f"{'Comparison':<30} "
            f"{'t-stat':<10} "
            f"{'p-value':<10} "
            f"{'p-corr':<10} "
            f"{'Sig':<6}"
        )
        report.append("-" * 70)
        
        for _, row in df_comparisons.iterrows():
            comparison = f"{row['method_a']} vs {row['method_b']}"
            sig = "*" if row['significant_corrected'] else "ns"
            report.append(
                f"{comparison:<30} "
                f"{row['t_statistic']:<10.3f} "
                f"{row['p_value']:<10.4f} "
                f"{row['p_value_corrected']:<10.4f} "
                f"{sig:<6}"
            )
        
        report.append("\n* p < 0.05 (Bonferroni corrected)")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_results(self, scores_dict: Dict[str, np.ndarray], output_path: str = None):
        """Save statistical analysis results to files."""
        from config.algorithms_config import get_timestamped_filename, get_results_path
        
        # Generate report
        report = self.generate_report(scores_dict)
        
        # Save report text file
        if output_path is None:
            report_path = get_results_path(get_timestamped_filename('statistical_analysis_report', 'txt'))
        else:
            base_name = output_path.replace('.txt', '')
            report_path = get_results_path(get_timestamped_filename(base_name, 'txt'))
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")
        
        # Save comparison data as CSV
        df_comparisons = self.compare_all_pairs(scores_dict)
        csv_path = report_path.replace('.txt', '.csv')
        df_comparisons.to_csv(csv_path, index=False)
        print(f"Comparison data saved to {csv_path}")
        
        # Save descriptive statistics as JSON
        descriptive_stats = {}
        for method, scores in scores_dict.items():
            descriptive_stats[method] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores, ddof=1)),
                'median': float(np.median(scores)),
                'n': len(scores),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        json_path = report_path.replace('.txt', '.json')
        import json
        with open(json_path, 'w') as f:
            json.dump(descriptive_stats, f, indent=2)
        print(f"Descriptive statistics saved to {json_path}")


def example_usage():
    """Example of how to use the statistical analyzer."""
    np.random.seed(42)
    
    # Simulate results from 10 subjects
    n_subjects = 10
    
    # Generate synthetic data with some effect
    baseline_scores = np.random.normal(0.65, 0.08, n_subjects)
    proposed_scores = baseline_scores + np.random.normal(0.08, 0.05, n_subjects)
    comparison_scores = baseline_scores + np.random.normal(0.03, 0.06, n_subjects)
    
    scores_dict = {
        'Classic_CSP': np.clip(baseline_scores, 0, 1),
        'FBCSP': np.clip(comparison_scores, 0, 1),
        'Proposed': np.clip(proposed_scores, 0, 1)
    }
    
    # Run analysis
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    print(analyzer.generate_report(scores_dict))
    
    # Save results
    analyzer.save_results(scores_dict)
    
    # Detailed comparison
    print("\n\nDetailed comparison: Proposed vs Classic_CSP")
    print("-" * 70)
    result = analyzer.paired_ttest(
        scores_dict['Proposed'], 
        scores_dict['Classic_CSP'],
        "Proposed", "Classic_CSP"
    )
    
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()
