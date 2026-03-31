"""
消融实验 - 在所有9个BCI IV 2A受试者上运行
生成论文Table IX所需的统计数据 (Mean ± Std)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.fbcsp import FilterBankCSPClassifier, FBCSPConfig
from algorithms_collection import FilterBankTangentSpace
from config import mi_config as cfg
from data_loader_moabb import load_bci_iv_2a_moabb


class AblationStudyAllSubjects:
    """在所有受试者上执行消融实验并汇总统计结果。"""
    
    def __init__(self):
        self.results_by_subject: Dict[int, Dict[str, Dict]] = {}
        self.aggregated_results: Dict[str, Dict] = {}
        
    def evaluate_full_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估完整模型 (Filter Bank + Tangent Space + SVM)。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_no_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估无特征选择版本 (使用所有特征)。"""
        # 使用一个很大的n_features值来禁用特征选择
        n_total_features = 9 * 22 * 23 // 2  # 9 bands * 22 channels * (22+1)/2
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=n_total_features,  # 使用所有特征
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_single_band(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估单频带版本 - 使用1个频带。"""
        model = FilterBankTangentSpace(
            n_bands=1,  # 单频带
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_3_bands(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估3频带版本。"""
        model = FilterBankTangentSpace(
            n_bands=3,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_5_bands(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估5频带版本。"""
        model = FilterBankTangentSpace(
            n_bands=5,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_no_shrinkage(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估无收缩估计版本 (使用SCM)。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='scm',  # 样本协方差矩阵，无收缩
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_euclidean(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估欧几里得度量版本 (不使用Riemannian Tangent Space)。"""
        # 使用FBCSP + SVM (无Riemannian)
        config = FBCSPConfig(
            sample_rate=cfg.SAMPLE_RATE_HZ,
            filter_banks=cfg.FILTER_BANKS,
            components_per_band=cfg.CSP_COMPONENTS_PER_BAND,
            svm_kernel=cfg.SVM_KERNEL,
            svm_c=cfg.SVM_C,
            use_riemann_tangent=False
        )
        model = FilterBankCSPClassifier(config)
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_lda(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估LDA分类器版本。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='lda',  # 使用LDA分类器
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_rf(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估随机森林分类器版本。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='rf',  # 使用随机森林分类器
            n_features=100,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_50_features(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估50个特征版本。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=50,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def evaluate_200_features(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估200个特征版本。"""
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=200,
            fs=cfg.SAMPLE_RATE_HZ
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': acc * 100,
            'train_time': train_time,
            'infer_time': infer_time,
            'success': True
        }
    
    def run_study_on_subject(self, subject_id: int) -> Dict[str, Dict]:
        """在单个受试者上运行完整消融实验。"""
        print(f"\n{'='*60}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*60}")
        
        # 加载数据
        try:
            X_train, y_train, _ = load_bci_iv_2a_moabb(
                subjects=[subject_id], 
                use_test_data=False
            )
            X_test, y_test, _ = load_bci_iv_2a_moabb(
                subjects=[subject_id], 
                use_test_data=True
            )
            
            print(f"Training data: {X_train.shape}, Classes: {np.unique(y_train)}")
            print(f"Test data: {X_test.shape}, Classes: {np.unique(y_test)}")
        except Exception as e:
            print(f"Error loading data for subject {subject_id}: {e}")
            return {}
        
        # 评估所有配置
        subject_results = {}
        
        print("\n  Evaluating configurations:")
        
        print("    1. Full Model")
        subject_results['Full_Model'] = self.evaluate_full_model(X_train, y_train, X_test, y_test)
        
        print("    2. No Feature Selection")
        subject_results['No_Feature_Selection'] = self.evaluate_no_feature_selection(X_train, y_train, X_test, y_test)
        
        print("    3. Single Band")
        subject_results['Single_Band'] = self.evaluate_single_band(X_train, y_train, X_test, y_test)
        
        print("    4. 3 Bands")
        subject_results['3_Bands'] = self.evaluate_3_bands(X_train, y_train, X_test, y_test)
        
        print("    5. 5 Bands")
        subject_results['5_Bands'] = self.evaluate_5_bands(X_train, y_train, X_test, y_test)
        
        print("    6. No Shrinkage (SCM)")
        subject_results['Estimator_SCM'] = self.evaluate_no_shrinkage(X_train, y_train, X_test, y_test)
        
        print("    7. Euclidean (No Riemannian)")
        subject_results['Metric_Euclid'] = self.evaluate_euclidean(X_train, y_train, X_test, y_test)
        
        print("    8. LDA Classifier")
        subject_results['Classifier_LDA'] = self.evaluate_lda(X_train, y_train, X_test, y_test)
        
        print("    9. RF Classifier")
        subject_results['Classifier_RF'] = self.evaluate_rf(X_train, y_train, X_test, y_test)
        
        print("    10. 50 Features")
        subject_results['50_Features'] = self.evaluate_50_features(X_train, y_train, X_test, y_test)
        
        print("    11. 200 Features")
        subject_results['200_Features'] = self.evaluate_200_features(X_train, y_train, X_test, y_test)
        
        return subject_results
    
    def run_study_all_subjects(self, subjects: List[int] = None):
        """在所有受试者上运行消融实验。"""
        if subjects is None:
            subjects = list(range(1, 10))  # BCI IV 2A有9个受试者
        
        print(f"\n{'='*70}")
        print(f"ABLATION STUDY - ALL SUBJECTS")
        print(f"{'='*70}")
        print(f"Subjects: {subjects}")
        print(f"Protocol: Session 1 (Train) -> Session 2 (Test)")
        
        # 在每个受试者上运行
        for subject_id in subjects:
            results = self.run_study_on_subject(subject_id)
            if results:
                self.results_by_subject[subject_id] = results
        
        # 汇总统计
        self.aggregate_results()
        
    def aggregate_results(self):
        """汇总所有受试者的结果，计算Mean ± Std。"""
        print(f"\n{'='*70}")
        print(f"AGGREGATING RESULTS ACROSS ALL SUBJECTS")
        print(f"{'='*70}")
        
        if not self.results_by_subject:
            print("No results to aggregate!")
            return
        
        # 获取所有配置名称
        config_names = list(next(iter(self.results_by_subject.values())).keys())
        
        for config_name in config_names:
            accuracies = []
            train_times = []
            
            for subject_id, results in self.results_by_subject.items():
                if config_name in results and results[config_name].get('success', False):
                    accuracies.append(results[config_name]['accuracy'])
                    train_times.append(results[config_name]['train_time'])
            
            if accuracies:
                self.aggregated_results[config_name] = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'train_time_mean': np.mean(train_times),
                    'n_subjects': len(accuracies)
                }
    
    def print_results(self):
        """打印汇总结果表格 - 对应论文Table IX格式。"""
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY RESULTS (Mean ± Std across 9 Subjects)")
        print(f"{'='*80}")
        print(f"{'Configuration':<25} {'Accuracy (%)':<20} {'Training Time (s)':<15}")
        print(f"{'-'*80}")
        
        # 定义显示顺序 (对应论文Table IX)
        display_order = [
            'Full_Model',
            'No_Feature_Selection',
            'Single_Band',
            '3_Bands',
            '5_Bands',
            'Estimator_SCM',
            'Metric_Euclid',
            'Classifier_LDA',
            'Classifier_RF',
            '50_Features',
            '200_Features'
        ]
        
        for config_name in display_order:
            if config_name in self.aggregated_results:
                result = self.aggregated_results[config_name]
                acc_str = f"{result['accuracy_mean']:.2f} ± {result['accuracy_std']:.1f}"
                time_str = f"{result['train_time_mean']:.2f}"
                print(f"{config_name:<25} {acc_str:<20} {time_str:<15}")
        
        print(f"{'-'*80}")
        
        # 打印相对改进
        if 'Full_Model' in self.aggregated_results:
            full_acc = self.aggregated_results['Full_Model']['accuracy_mean']
            
            if 'Single_Band' in self.aggregated_results:
                single_acc = self.aggregated_results['Single_Band']['accuracy_mean']
                fb_improvement = full_acc - single_acc
                print(f"\nFilter Bank contribution: +{fb_improvement:.2f}%")
            
            if 'Metric_Euclid' in self.aggregated_results:
                euclid_acc = self.aggregated_results['Metric_Euclid']['accuracy_mean']
                riemann_improvement = full_acc - euclid_acc
                print(f"Riemannian metric contribution: +{riemann_improvement:.2f}%")
            
            if 'No_Feature_Selection' in self.aggregated_results:
                no_fs_acc = self.aggregated_results['No_Feature_Selection']['accuracy_mean']
                fs_improvement = full_acc - no_fs_acc
                print(f"Feature selection contribution: +{fs_improvement:.2f}%")
    
    def save_results(self, output_dir: str = None):
        """保存结果到CSV和JSON。"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存每个受试者的详细结果
        detailed_data = []
        for subject_id, results in self.results_by_subject.items():
            for config_name, result in results.items():
                if result.get('success', False):
                    detailed_data.append({
                        'subject_id': subject_id,
                        'config_name': config_name,
                        'accuracy': result['accuracy'],
                        'train_time': result['train_time'],
                        'infer_time': result['infer_time']
                    })
        
        df_detailed = pd.DataFrame(detailed_data)
        detailed_path = output_dir / f'ablation_detailed_{timestamp}.csv'
        df_detailed.to_csv(detailed_path, index=False)
        print(f"\nDetailed results saved to: {detailed_path}")
        
        # 保存汇总结果 (Table IX格式)
        summary_data = []
        for config_name, result in self.aggregated_results.items():
            summary_data.append({
                'Configuration': config_name,
                'Accuracy_Mean': result['accuracy_mean'],
                'Accuracy_Std': result['accuracy_std'],
                'Training_Time_Mean': result['train_time_mean'],
                'N_Subjects': result['n_subjects']
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = output_dir / f'ablation_summary_{timestamp}.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary results saved to: {summary_path}")
        
        # 保存JSON格式
        json_path = output_dir / f'ablation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'aggregated': self.aggregated_results,
                'by_subject': {
                    str(k): v for k, v in self.results_by_subject.items()
                }
            }, f, indent=2)
        print(f"JSON results saved to: {json_path}")
        
        return summary_path


def main():
    """主函数 - 在所有9个受试者上运行消融实验。"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ablation study on all 9 BCI IV 2A subjects"
    )
    parser.add_argument(
        '--subjects', 
        nargs='+', 
        type=int, 
        default=None,
        help='Subject IDs to process (default: all 9 subjects)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for results'
    )
    args = parser.parse_args()
    
    # 运行消融实验
    study = AblationStudyAllSubjects()
    study.run_study_all_subjects(subjects=args.subjects)
    
    # 打印结果
    study.print_results()
    
    # 保存结果
    study.save_results(output_dir=args.output_dir)
    
    print(f"\n{'='*70}")
    print("Ablation study completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
