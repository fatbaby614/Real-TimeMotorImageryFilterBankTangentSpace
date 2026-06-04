#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
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


def evaluate_config(subject_id, config_name, config_data, dataset='BCI_IV_2A'):
    """评估特定电极配置（cross-session协议：Session 1训练 → Session 2测试）"""
    print(f"\n{'=' * 80}")
    print(f"Evaluating Subject {subject_id} - {config_name}: {config_data['description']}")
    print(f"{'=' * 80}")
    print("  Cross-session protocol: Session 1 -> Session 2")
    
    # 加载 Session 1（训练）和 Session 2（测试）
    X_train, y_train, _ = data_loader.load_single_subject_moabb(
        subject_id, use_test_data=False, dataset=dataset
    )
    X_test, y_test, _ = data_loader.load_single_subject_moabb(
        subject_id, use_test_data=True, dataset=dataset
    )
    
    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape:     {X_test.shape}")
    
    # 获取通道索引
    channel_indices = get_channel_indices(BCI_IV_2A_CHANNELS, config_data['channels'])
    print(f"Selected channels: {config_data['channels'] if config_data['channels'] else 'All channels'}")
    print(f"Channel indices: {channel_indices}")
    
    # 选择指定通道（训练集和测试集都选）
    X_train_selected = select_channels(X_train, channel_indices)
    X_test_selected = select_channels(X_test, channel_indices)
    print(f"Selected training data shape: {X_train_selected.shape}")
    print(f"Selected test data shape:     {X_test_selected.shape}")
    
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
    
    model.fit(X_train_selected, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\n  Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}, Time: {train_time:.2f}s")
    
    results = {
        'subject_id': subject_id,
        'dataset': dataset,
        'config_name': config_name,
        'description': config_data['description'],
        'channels': str(config_data['channels']) if config_data['channels'] else 'All',
        'n_channels': len(channel_indices),
        'accuracy': accuracy,
        'kappa': kappa,
        'train_time': train_time
    }
    
    print(f"\n{config_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}, Time: {train_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare FilterBankTangentSpace+SVM performance with different channel configurations")
    parser.add_argument("--subjects", type=str, nargs='*', default=['1~9'],
                        help="Subject IDs/range to evaluate (e.g., '1~9', '1 2 3', default: '1~9')")
    parser.add_argument("--configs", type=str, nargs='+', default=list(CHANNEL_CONFIGS.keys()),
                        help=f"Channel configurations to test (default: all)")
    parser.add_argument("--dataset", type=str, default='BCI_IV_2A',
                        help="Dataset to use (default: BCI_IV_2A)")
    parser.add_argument("--output", type=str, default='results/channel_comparison_detailed.csv',
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FilterBankTangentSpace+SVM Channel Configuration Comparison")
    print("Protocol: Cross-session (Session 1 -> Session 2)")
    print("=" * 80)
    
    # 验证配置名称
    valid_configs = [cfg for cfg in args.configs if cfg in CHANNEL_CONFIGS]
    if not valid_configs:
        print("Error: No valid channel configurations specified!")
        print(f"Available configurations: {list(CHANNEL_CONFIGS.keys())}")
        return
    
    # 解析受试者范围
    subjects_str = ' '.join(args.subjects) if isinstance(args.subjects, list) else args.subjects
    subjects = []
    for token in subjects_str.replace(',', ' ').split():
        if '~' in token:
            start, end = map(int, token.split('~'))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(token))
    subjects = sorted(set(subjects))
    
    # 评估所有配置
    all_results = []
    
    for subject_id in subjects:
        for config_name in valid_configs:
            results = evaluate_config(
                subject_id=subject_id,
                config_name=config_name,
                config_data=CHANNEL_CONFIGS[config_name],
                dataset=args.dataset
            )
            all_results.append(results)
    
    # 创建结果DataFrame（与主评估格式一致）
    df_results = pd.DataFrame([{
        'dataset': r['dataset'],
        'algorithm': 'FilterBankTangentSpace+SVM',
        'subject': r['subject_id'],
        'config_name': r['config_name'],
        'description': r['description'],
        'channels': r['channels'],
        'n_channels': r['n_channels'],
        'accuracy': r['accuracy'],
        'kappa': r['kappa'],
        'train_time': r['train_time']
    } for r in all_results])
    
    # 按通道数量排序
    df_results = df_results.sort_values('n_channels')
    
    # 保存结果
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # 打印对比结果（按subject汇总）
    print("\n" + "=" * 80)
    print("Channel Configuration Comparison Results (Cross-Session: Session 1 -> Session 2)")
    print("=" * 80)
    
    # 按配置分组显示汇总统计
    print(f"\n{'Config':<20} {'Ch':<5} {'Accuracy (Mean±Std)':<25} {'Kappa (Mean±Std)':<25} {'Time (s)':<10}")
    print("-" * 90)
    
    for config_name in valid_configs:
        config_df = df_results[df_results['config_name'] == config_name]
        if len(config_df) > 0:
            mean_acc = config_df['accuracy'].mean()
            std_acc = config_df['accuracy'].std()
            mean_kap = config_df['kappa'].mean()
            std_kap = config_df['kappa'].std()
            mean_time = config_df['train_time'].mean()
            n_ch = config_df['n_channels'].iloc[0]
            print(f"{config_name:<20} {n_ch:<5} {mean_acc:.4f} ± {std_acc:.4f}         {mean_kap:.4f} ± {std_kap:.4f}         {mean_time:.2f}")
    
    # 计算相对于全通道的性能下降
    all_channels_df = df_results[df_results['config_name'] == 'all_channels']
    if len(all_channels_df) > 0:
        all_channels_acc = all_channels_df['accuracy'].mean()
        print(f"\nPerformance relative to all channels ({all_channels_acc:.4f}):")
        for config_name in valid_configs:
            if config_name != 'all_channels':
                config_df = df_results[df_results['config_name'] == config_name]
                config_acc = config_df['accuracy'].mean()
                performance_ratio = (config_acc / all_channels_acc) * 100
                performance_drop = all_channels_acc - config_acc
                print(f"  {config_name:<20}: {performance_ratio:.1f}% ({performance_drop:+.4f} vs all channels)")


if __name__ == "__main__":
    main()