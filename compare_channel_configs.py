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
from sklearn.metrics import accuracy_score, cohen_kappa_score
from config.algorithms_config import RANDOM_STATE, N_SPLITS
from algorithms_collection import FilterBankTangentSpace
import data_loader_moabb as data_loader


# 不同的电极配置
CHANNEL_CONFIGS = {
    "all_channels": {
        "description": "所有22个电极",
        "channels": None  # 使用所有通道
    },
    "motor_core_8": {
        "description": "运动核心区8电极",
        "channels": ["C3", "C4", "Cz", "FC1", "FC2", "FCz", "CP1", "CP2"]
    },
    "motor_core_6": {
        "description": "运动核心区6电极",
        "channels": ["C3", "C4", "Cz", "FC1", "FC2", "FCz"]
    },
    "minimal_4": {
        "description": "最少4电极",
        "channels": ["C3", "C4", "Cz", "FCz"]
    },
    "sensorimotor_10": {
        "description": "感觉运动区10电极",
        "channels": ["C3", "C4", "Cz", "FC1", "FC2", "FCz", "CP1", "CP2", "C1", "C2"]
    },
    "extended_12": {
        "description": "扩展12电极",
        "channels": ["C3", "C4", "Cz", "FC1", "FC2", "FCz", "CP1", "CP2", "C1", "C2", "Fz", "Pz"]
    }
}

# BCI IV 2A 数据集的22个电极名称
BCI_IV_2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]


def get_channel_indices(channel_names, selected_channels):
    """获取选中通道的索引"""
    if selected_channels is None:
        return list(range(len(channel_names)))
    
    indices = []
    for ch in selected_channels:
        if ch in channel_names:
            indices.append(channel_names.index(ch))
        else:
            print(f"Warning: Channel {ch} not found in data")
    return indices


def select_channels(X, channel_indices):
    """选择特定通道的数据"""
    return X[:, channel_indices, :]


def evaluate_config(subject_id, config_name, config_data, use_test_data=False, dataset='BCI_IV_2A'):
    """评估特定电极配置"""
    print(f"\n{'=' * 80}")
    print(f"Evaluating Subject {subject_id} - {config_name}: {config_data['description']}")
    print(f"{'=' * 80}")
    
    X, y, meta = data_loader.load_single_subject_moabb(subject_id, use_test_data=use_test_data, dataset=dataset)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\nOriginal data shape: {X.shape}")
    
    # 获取通道索引
    channel_indices = get_channel_indices(BCI_IV_2A_CHANNELS, config_data['channels'])
    print(f"Selected channels: {config_data['channels'] if config_data['channels'] else 'All channels'}")
    print(f"Channel indices: {channel_indices}")
    
    # 选择指定通道
    X_selected = select_channels(X, channel_indices)
    print(f"Selected data shape: {X_selected.shape}")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_accuracies = []
    fold_kappas = []
    fold_times = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
        print(f"\nFold {fold_idx + 1}/{N_SPLITS}")
        
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练FilterBankTangentSpace+SVM
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
        
        fold_accuracies.append(accuracy)
        fold_kappas.append(kappa)
        fold_times.append(train_time)
        
        print(f"  Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}, Time: {train_time:.2f}s")
    
    results = {
        'subject_id': subject_id,
        'dataset': dataset,
        'config_name': config_name,
        'description': config_data['description'],
        'channels': str(config_data['channels']) if config_data['channels'] else 'All',
        'n_channels': len(channel_indices),
        'fold_accuracies': fold_accuracies,
        'fold_kappas': fold_kappas,
        'fold_times': fold_times,
        'mean_accuracy': np.mean(fold_accuracies),
        'std_accuracy': np.std(fold_accuracies),
        'min_accuracy': np.min(fold_accuracies),
        'max_accuracy': np.max(fold_accuracies),
        'mean_kappa': np.mean(fold_kappas),
        'std_kappa': np.std(fold_kappas),
        'min_kappa': np.min(fold_kappas),
        'max_kappa': np.max(fold_kappas),
        'mean_train_time': np.mean(fold_times),
        'std_train_time': np.std(fold_times)
    }
    
    print(f"\n{config_name} Results:")
    print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f} (Min: {results['min_accuracy']:.4f}, Max: {results['max_accuracy']:.4f})")
    print(f"  Mean Kappa: {results['mean_kappa']:.4f} ± {results['std_kappa']:.4f} (Min: {results['min_kappa']:.4f}, Max: {results['max_kappa']:.4f})")
    print(f"  Mean Time: {results['mean_train_time']:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare FilterBankTangentSpace+SVM performance with different channel configurations")
    parser.add_argument("--subjects", type=int, nargs='+', default=[1],
                        help="Subject IDs to evaluate (default: [1])")
    parser.add_argument("--configs", type=str, nargs='+', default=list(CHANNEL_CONFIGS.keys()),
                        help=f"Channel configurations to test (default: all)")
    parser.add_argument("--use-test-data", action="store_true",
                        help="Use test data instead of training data")
    parser.add_argument("--dataset", type=str, default='BCI_IV_2A',
                        help="Dataset to use (default: BCI_IV_2A)")
    parser.add_argument("--output", type=str, default='results/channel_comparison_detailed.csv',
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FilterBankTangentSpace+SVM Channel Configuration Comparison")
    print("=" * 80)
    
    # 验证配置名称
    valid_configs = [cfg for cfg in args.configs if cfg in CHANNEL_CONFIGS]
    if not valid_configs:
        print("Error: No valid channel configurations specified!")
        print(f"Available configurations: {list(CHANNEL_CONFIGS.keys())}")
        return
    
    # 评估所有配置
    all_results = []
    
    for subject_id in args.subjects:
        for config_name in valid_configs:
            results = evaluate_config(
                subject_id=subject_id,
                config_name=config_name,
                config_data=CHANNEL_CONFIGS[config_name],
                use_test_data=args.use_test_data,
                dataset=args.dataset
            )
            all_results.append(results)
    
    # 创建结果DataFrame
    df_results = pd.DataFrame([{
        'dataset': r['dataset'],
        'algorithm': 'FilterBankTangentSpace+SVM',
        'subject': r['subject_id'],
        'config_name': r['config_name'],
        'description': r['description'],
        'channels': r['channels'],
        'n_channels': r['n_channels'],
        'accuracy_mean': r['mean_accuracy'],
        'accuracy_std': r['std_accuracy'],
        'accuracy_min': r['min_accuracy'],
        'accuracy_max': r['max_accuracy'],
        'kappa_mean': r['mean_kappa'],
        'kappa_std': r['std_kappa'],
        'kappa_min': r['min_kappa'],
        'kappa_max': r['max_kappa'],
        'train_time_mean': r['mean_train_time'],
        'train_time_std': r['std_train_time']
    } for r in all_results])
    
    # 按通道数量排序
    df_results = df_results.sort_values('n_channels')
    
    # 重新排列列的顺序，与标准格式一致
    df_results = df_results[[
        'dataset', 'algorithm', 'subject', 'config_name', 'description', 
        'channels', 'n_channels', 'accuracy_mean', 'accuracy_std', 
        'accuracy_min', 'accuracy_max', 'kappa_mean', 'kappa_std', 
        'kappa_min', 'kappa_max', 'train_time_mean', 'train_time_std'
    ]]
    
    # 保存结果
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("Channel Configuration Comparison Results")
    print("=" * 80)
    
    # 按配置分组显示结果
    for config_name in valid_configs:
        config_df = df_results[df_results['config_name'] == config_name]
        if len(config_df) > 0:
            print(f"\n{config_name} ({config_df.iloc[0]['description']}):")
            print(f"  Channels: {config_df.iloc[0]['n_channels']}")
            print(f"  Accuracy: {config_df['accuracy_mean'].mean():.4f} ± {config_df['accuracy_mean'].std():.4f} (Min: {config_df['accuracy_min'].min():.4f}, Max: {config_df['accuracy_max'].max():.4f})")
            print(f"  Kappa: {config_df['kappa_mean'].mean():.4f} ± {config_df['kappa_mean'].std():.4f} (Min: {config_df['kappa_min'].min():.4f}, Max: {config_df['kappa_max'].max():.4f})")
            print(f"  Time: {config_df['train_time_mean'].mean():.2f}s ± {config_df['train_time_mean'].std():.2f}s")
    
    # 性能对比表格
    print("\n" + "=" * 80)
    print("Performance Comparison Table")
    print("=" * 80)
    summary_df = df_results.groupby('config_name').agg({
        'n_channels': 'first',
        'accuracy_mean': ['mean', 'std'],
        'kappa_mean': ['mean', 'std'],
        'train_time_mean': ['mean', 'std']
    }).round(4)
    print(summary_df.to_string())
    
    # 计算相对于全通道的性能下降
    all_channels_acc = df_results[df_results['config_name'] == 'all_channels']['accuracy_mean'].mean()
    if all_channels_acc > 0:
        print(f"\nPerformance relative to all channels ({all_channels_acc:.4f}):")
        for config_name in valid_configs:
            if config_name != 'all_channels':
                config_acc = df_results[df_results['config_name'] == config_name]['accuracy_mean'].mean()
                performance_ratio = (config_acc / all_channels_acc) * 100
                performance_drop = all_channels_acc - config_acc
                print(f"  {config_name}: {performance_ratio:.1f}% ({performance_drop:+.4f} accuracy)")


if __name__ == "__main__":
    main()