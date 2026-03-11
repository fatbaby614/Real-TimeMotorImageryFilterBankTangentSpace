#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from config.algorithms_config import RANDOM_STATE, N_SPLITS
from algorithms_collection import FilterBankTangentSpace
import data_loader_moabb as data_loader


# 8个电极配置（BCI IV 2A数据集中可用的通道）
# 选择对运动想象最重要的通道：传感器运动皮层和运动前区
SELECTED_CHANNELS_8 = ["C3", "C4", "Cz", "FC1", "FC2", "FCz", "CP1", "CP2"]

# BCI IV 2A 数据集的22个电极名称（标准10-20系统）
BCI_IV_2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]


def get_channel_indices(channel_names, selected_channels):
    """
    获取选中通道的索引
    
    Parameters:
    -----------
    channel_names : list
        所有通道名称
    selected_channels : list
        要选择的通道名称
        
    Returns:
    --------
    indices : list
        选中通道的索引列表
    """
    indices = []
    for ch in selected_channels:
        if ch in channel_names:
            indices.append(channel_names.index(ch))
        else:
            print(f"Warning: Channel {ch} not found in data")
    return indices


def select_channels(X, channel_indices):
    """
    选择特定通道的数据
    
    Parameters:
    -----------
    X : np.ndarray
        输入数据，形状 (n_samples, n_channels, n_times)
    channel_indices : list
        要选择的通道索引
        
    Returns:
    --------
    X_selected : np.ndarray
        选择后的数据，形状 (n_samples, len(channel_indices), n_times)
    """
    return X[:, channel_indices, :]


def evaluate_with_channels(subject_id, channel_indices, channel_names, 
                         use_test_data=False, dataset='BCI_IV_2A'):
    """
    使用指定通道评估FilterBankTangentSpace+SVM算法
    
    Parameters:
    -----------
    subject_id : int
        受试者ID
    channel_indices : list
        要使用的通道索引
    channel_names : list
        通道名称列表
    use_test_data : bool
        是否使用测试数据
    dataset : str
        数据集名称
        
    Returns:
    --------
    results : dict
        评估结果
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating Subject {subject_id} on {dataset}")
    print(f"Using channels: {channel_names}")
    print(f"Channel indices: {channel_indices}")
    print(f"{'=' * 80}")
    
    X, y, meta = data_loader.load_single_subject_moabb(subject_id, use_test_data=use_test_data, dataset=dataset)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\nOriginal data shape: {X.shape}")
    
    # 选择指定通道
    X_selected = select_channels(X, channel_indices)
    print(f"Selected data shape: {X_selected.shape}")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_accuracies = []
    fold_kappas = []
    fold_times = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{N_SPLITS}")
        print(f"{'=' * 60}")
        
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练FilterBankTangentSpace+SVM
        print("Training FilterBankTangentSpace+SVM...")
        import time
        start_time = time.time()
        
        model = FilterBankTangentSpace(
            n_bands=9,
            estimator='oas',
            metric='riemann',
            classifier='svm',
            n_features=100,
            fs=250
        )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        fold_accuracies.append(accuracy)
        fold_kappas.append(kappa)
        fold_times.append(train_time)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Kappa: {kappa:.4f}")
        print(f"Training time: {train_time:.2f}s")
        print(f"Confusion Matrix:\n{cm}")
    
    results = {
        'subject_id': subject_id,
        'dataset': dataset,
        'channels': channel_names,
        'n_channels': len(channel_indices),
        'channel_indices': channel_indices,
        'fold_accuracies': fold_accuracies,
        'fold_kappas': fold_kappas,
        'fold_times': fold_times,
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'mean_kappa': np.mean(fold_kappas),
        'std_kappa': np.std(fold_kappas),
        'mean_train_time': np.mean(fold_times),
        'std_train_time': np.std(fold_times)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FilterBankTangentSpace+SVM with specific channels")
    parser.add_argument("--subjects", type=int, nargs='+', default=list(range(1, 10)),
                        help="Subject IDs to evaluate (default: 1-9)")
    parser.add_argument("--channels", type=str, nargs='+', default=SELECTED_CHANNELS_8,
                        help=f"Channel names to use (default: {SELECTED_CHANNELS_8})")
    parser.add_argument("--use-test-data", action="store_true",
                        help="Use test data instead of training data")
    parser.add_argument("--dataset", type=str, default='BCI_IV_2A',
                        help="Dataset to use (default: BCI_IV_2A)")
    parser.add_argument("--output", type=str, default='results/channel_comparison_results.csv',
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FilterBankTangentSpace+SVM Channel Comparison Evaluation")
    print("=" * 80)
    
    # 获取通道索引
    channel_indices = get_channel_indices(BCI_IV_2A_CHANNELS, args.channels)
    
    if len(channel_indices) != len(args.channels):
        print(f"Warning: Only {len(channel_indices)}/{len(args.channels)} channels found")
        print(f"Available channels: {BCI_IV_2A_CHANNELS}")
    
    if len(channel_indices) == 0:
        print("Error: No valid channels found!")
        return
    
    # 评估所有受试者
    all_results = []
    
    for subject_id in args.subjects:
        results = evaluate_with_channels(
            subject_id=subject_id,
            channel_indices=channel_indices,
            channel_names=args.channels,
            use_test_data=args.use_test_data,
            dataset=args.dataset
        )
        all_results.append(results)
    
    # 创建结果DataFrame
    df_results = pd.DataFrame([{
        'subject_id': r['subject_id'],
        'dataset': r['dataset'],
        'channels': ','.join(r['channels']),
        'n_channels': r['n_channels'],
        'mean_accuracy': r['mean_accuracy'],
        'std_accuracy': r['std_accuracy'],
        'mean_kappa': r['mean_kappa'],
        'std_kappa': r['std_kappa'],
        'mean_train_time': r['mean_train_time'],
        'std_train_time': r['std_train_time']
    } for r in all_results])
    
    # 计算总体统计
    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)
    print(f"Channels: {args.channels}")
    print(f"Number of subjects: {len(all_results)}")
    print(f"\nAccuracy:")
    print(f"  Mean: {df_results['mean_accuracy'].mean():.4f} ± {df_results['mean_accuracy'].std():.4f}")
    print(f"  Range: [{df_results['mean_accuracy'].min():.4f}, {df_results['mean_accuracy'].max():.4f}]")
    print(f"\nKappa:")
    print(f"  Mean: {df_results['mean_kappa'].mean():.4f} ± {df_results['mean_kappa'].std():.4f}")
    print(f"  Range: [{df_results['mean_kappa'].min():.4f}, {df_results['mean_kappa'].max():.4f}]")
    print(f"\nTraining time:")
    print(f"  Mean: {df_results['mean_train_time'].mean():.2f}s ± {df_results['mean_train_time'].std():.2f}s")
    
    # 保存结果
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # 打印详细结果
    print("\n" + "=" * 80)
    print("Detailed Results by Subject")
    print("=" * 80)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()