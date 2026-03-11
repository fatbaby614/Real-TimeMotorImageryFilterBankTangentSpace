#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import os
from config.algorithms_config import RESULTS_PATH, RANDOM_STATE, N_SPLITS, get_timestamped_filename, get_results_path
from algorithms_collection import get_algorithm
import data_loader_moabb as data_loader
import visualization
import torch


def extract_features_from_model(model, X, algo_name):
    """
    Extract features from a trained model for t-SNE visualization
    
    Parameters:
    -----------
    model : trained model object
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_times)
    algo_name : str
        Name of the algorithm
        
    Returns:
    --------
    features : np.ndarray
        Extracted features of shape (n_samples, n_features)
    """
    try:
        if algo_name in ['CSP+LDA', 'CSP+SVM']:
            # Extract CSP features from the pipeline
            csp_transformer = model.pipeline.named_steps['csp']
            features = csp_transformer.transform(X)
            
        elif algo_name == 'FBCSP':
            # Extract FBCSP features
            features_list = []
            for i, (low, high) in enumerate(model.freq_bands):
                X_band = np.array([apply_bandpass_filter(trial, low, high, model.fs) for trial in X])
                csp_features = model.csp_transformers[i].transform(X_band)
                features_list.append(csp_features)
            features = np.hstack(features_list)
            
        elif algo_name in ['FilterBankTangentSpace', 'FilterBankTangentSpace+SVM', 
                          'FilterBankTangentSpace+LDA', 'FilterBankTangentSpace+RF']:
            # Extract Filter Bank Tangent Space features
            features_list = []
            for i, (low, high) in enumerate(model.freq_bands):
                X_band = np.array([apply_bandpass_filter(trial, low, high, model.fs) for trial in X])
                cov_matrices = model.cov_estimators[i].transform(X_band)
                ts_features = model.ts_transformers[i].transform(cov_matrices)
                features_list.append(ts_features)
            X_combined = np.hstack(features_list)
            features = model.feature_selector.transform(X_combined)
            
        elif algo_name == 'MDM':
            # Extract MDM features (covariance matrices)
            from pyriemann.estimation import Covariances
            cov_estimator = Covariances(estimator=model.estimator)
            cov_matrices = cov_estimator.fit_transform(X)
            # Flatten covariance matrices to 2D for t-SNE
            n_samples = cov_matrices.shape[0]
            n_channels = cov_matrices.shape[1]
            features = cov_matrices.reshape(n_samples, -1)
            
        elif algo_name in ['RiemannTangentSpace', 'RiemannTangentSpace+SVM', 
                          'RiemannTangentSpace+RF', 'RiemannTangentSpace+PCA']:
            # Extract tangent space features
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
            cov_estimator = Covariances(estimator=model.estimator)
            cov_matrices = cov_estimator.fit_transform(X)
            ts_transformer = TangentSpace(metric=model.metric)
            features = ts_transformer.fit_transform(cov_matrices)
            
        elif algo_name in ['EEGNet', 'EEGNex', 'EEG-Inception', 'ShallowFBCSPNet', 
                          'MSVTNet', 'IFNet', 'EEGConformer', 'CTNet', 'ATCNet', 'EEGSimpleConv',
                          'EEGTCNet', 'SincShallowNet', 'EEGITNet']:
            # Extract features from deep learning models
            model.model.eval()
            
            with torch.no_grad():
                # Preprocess input
                if hasattr(model, 'scaler'):
                    X_scaled = model.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
                else:
                    X_scaled = X
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_scaled).to(model.device)
                
                # Method 1: Extract intermediate layer features (better for t-SNE)
                try:
                    # Hook to capture features from the penultimate layer
                    intermediate_features = []
                    
                    def hook_fn(module, input, output):
                        intermediate_features.append(output)
                    
                    # Find the last linear layer or flatten layer
                    hook_handle = None
                    for name, module in reversed(list(model.model.named_modules())):
                        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Flatten):
                            hook_handle = module.register_forward_hook(hook_fn)
                            break
                    
                    if hook_handle is not None:
                        # Forward pass to capture intermediate features
                        if algo_name == 'EEGNet':
                            X_4d = X_tensor[:, :, :, None]
                            _ = model.model(X_4d)
                        else:
                            _ = model.model(X_tensor)
                        
                        # Get the captured features
                        if len(intermediate_features) > 0:
                            features = intermediate_features[0].cpu().numpy()
                            # Flatten if necessary
                            if len(features.shape) > 2:
                                features = features.reshape(features.shape[0], -1)
                            hook_handle.remove()
                        else:
                            # Fallback to predict_proba
                            if algo_name == 'EEGNet':
                                X_4d = X_tensor[:, :, :, None]
                                features = model.clf.predict_proba(X_4d)
                            else:
                                features = model.clf.predict_proba(X_tensor)
                    else:
                        # Fallback to predict_proba
                        if algo_name == 'EEGNet':
                            X_4d = X_tensor[:, :, :, None]
                            features = model.clf.predict_proba(X_4d)
                        else:
                            features = model.clf.predict_proba(X_tensor)
                except Exception as e:
                    # Fallback to predict_proba if intermediate extraction fails
                    if algo_name == 'EEGNet':
                        X_4d = X_tensor[:, :, :, None]
                        features = model.clf.predict_proba(X_4d)
                    else:
                        features = model.clf.predict_proba(X_tensor)
        
        else:
            # For other algorithms, try to get predict_proba
            if hasattr(model, 'predict_proba'):
                features = model.predict_proba(X)
            else:
                # Fallback: use flattened input as features
                features = X.reshape(X.shape[0], -1)
        
        return features
        
    except Exception as e:
        print(f"    Warning: Could not extract features for {algo_name}: {e}")
        # Fallback: use flattened input as features
        return X.reshape(X.shape[0], -1)


def apply_bandpass_filter(data, low_freq, high_freq, fs):
    from scipy import signal
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


def check_gpu():
    print("=" * 80)
    print("GPU Check")
    print("=" * 80)
    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"3. CUDA version: {torch.version.cuda}")
        print(f"4. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(f"     Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"     Compute capability: {torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}")
            print(f"     Multi-processor count: {torch.cuda.get_device_properties(i).multi_processor_count}")
        
        print(f"\n5. Current device: {torch.cuda.current_device()}")
        print(f"   Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        print("\nTesting GPU computation...")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("✅ GPU computation test successful!")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    print("\n" + "=" * 80)


def evaluate_subject(subject_id, algorithms, use_test_data=False, dataset='BCI_IV_2A', epochs=300, save_model=False, load_model=False, model_dir='models', extract_features=False):
    """评估单个受试者"""
    print(f"\n{'=' * 80}")
    print(f"Evaluating Subject {subject_id} on {dataset}")
    print(f"{'=' * 80}")
    
    X, y, meta = data_loader.load_single_subject_moabb(subject_id, use_test_data=use_test_data, dataset=dataset)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\nData loaded:")
    print(f"  Samples: {n_samples}")
    print(f"  Channels: {n_channels}")
    print(f"  Time points: {n_times}")
    print(f"  Classes: {n_classes}")
    
    results = []
    features_dict = {} if extract_features else None
    
    for algo_name in algorithms:
        print(f"\n{'=' * 60}")
        print(f"Evaluating algorithm: {algo_name}")
        print(f"{'=' * 60}")
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 模型保存/加载路径
            model_path = os.path.join(model_dir, f'{dataset}_subject{subject_id}_{algo_name}_fold{fold_idx+1}.pt')
            
            if load_model and os.path.exists(model_path):
                print(f"  Loading model from {model_path}")
                # 根据算法名称选择相应的加载方法
                if algo_name == 'CSP+LDA':
                    from algorithms_collection import CSPLDA
                    model = CSPLDA.load_model(model_path)
                elif algo_name == 'CSP+SVM':
                    from algorithms_collection import CSPSVM
                    model = CSPSVM.load_model(model_path)
                elif algo_name == 'FBCSP':
                    from algorithms_collection import FBCSP
                    model = FBCSP.load_model(model_path)
                elif algo_name in ['FilterBankTangentSpace', 'FilterBankTangentSpace+SVM', 
                                  'FilterBankTangentSpace+LDA', 'FilterBankTangentSpace+RF']:
                    from algorithms_collection import FilterBankTangentSpace
                    model = FilterBankTangentSpace.load_model(model_path)
                elif algo_name == 'MDM':
                    from algorithms_collection import MDM
                    model = MDM.load_model(model_path)
                elif algo_name == 'RiemannTangentSpace':
                    from algorithms_collection import RiemannTangentSpace
                    model = RiemannTangentSpace.load_model(model_path)
                elif algo_name == 'EEGNet':
                    from algorithms_collection import EEGNet
                    model = EEGNet.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEGNex':
                    from algorithms_collection import EEGNexClassifier
                    model = EEGNexClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEG-Inception':
                    from algorithms_collection import EEGInceptionClassifier
                    model = EEGInceptionClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'ShallowFBCSPNet':
                    from algorithms_collection import ShallowFBCSPNetClassifier
                    model = ShallowFBCSPNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'MSVTNet':
                    from algorithms_collection import MSVTNetClassifier
                    model = MSVTNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'IFNet':
                    from algorithms_collection import IFNetClassifier
                    model = IFNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEGConformer':
                    from algorithms_collection import EEGConformerClassifier
                    model = EEGConformerClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'CTNet':
                    from algorithms_collection import CTNetClassifier
                    model = CTNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'ATCNet':
                    from algorithms_collection import ATCNetClassifier
                    model = ATCNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEGSimpleConv':
                    from algorithms_collection import EEGSimpleConvClassifier
                    model = EEGSimpleConvClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEGTCNet':
                    from algorithms_collection import EEGTCNetClassifier
                    model = EEGTCNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'SincShallowNet':
                    from algorithms_collection import SincShallowNetClassifier
                    model = SincShallowNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                elif algo_name == 'EEGITNet':
                    from algorithms_collection import EEGITNetClassifier
                    model = EEGITNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
                else:
                    model = get_algorithm(algo_name, n_channels, n_times, n_classes)
                    # 检查模型的fit方法是否接受epochs参数
                    import inspect
                    fit_signature = inspect.signature(model.fit)
                    if 'epochs' in fit_signature.parameters:
                        model.fit(X_train, y_train, epochs=epochs)
                    else:
                        model.fit(X_train, y_train)
                train_time = 0
            else:
                model = get_algorithm(algo_name, n_channels, n_times, n_classes)
                
                start_time = time.time()
                # 检查模型的fit方法是否接受epochs参数
                import inspect
                fit_signature = inspect.signature(model.fit)
                if 'epochs' in fit_signature.parameters:
                    model.fit(X_train, y_train, epochs=epochs)
                else:
                    model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # 保存模型
                if save_model:
                    os.makedirs(model_dir, exist_ok=True)
                    print(f"  Saving model to {model_path}")
                    model.save_model(model_path)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes))).tolist()
            
            print(f"  Fold {fold_idx + 1}/{N_SPLITS}")
            print(f"    Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}")
            
            results.append({
                'dataset': dataset,
                'algorithm': algo_name,
                'subject': subject_id,
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'kappa': kappa,
                'train_time': train_time,
                'confusion_matrix': cm
            })
            
            # Extract features for t-SNE visualization (only for the first fold)
            if extract_features and fold_idx == 0:
                print(f"  Extracting features for t-SNE visualization...")
                features = extract_features_from_model(model, X, algo_name)
                features_dict[algo_name] = features
    
    # 确保返回的是正确的格式
    if extract_features:
        return results, features_dict
    else:
        return results


def generate_summary(results):
    """生成结果汇总"""
    df = pd.DataFrame(results)
    
    summary = df.groupby(['dataset', 'algorithm']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'kappa': ['mean', 'std', 'min', 'max'],
        'train_time': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={'algorithm_': 'algorithm'})
    
    return summary


def generate_subject_summary(results):
    """生成每个subject的结果汇总"""
    df = pd.DataFrame(results)
    
    subject_summary = df.groupby(['dataset', 'algorithm', 'subject']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'kappa': ['mean', 'std', 'min', 'max'],
        'train_time': ['mean', 'std']
    }).reset_index()
    
    subject_summary.columns = ['_'.join(col).strip('_') for col in subject_summary.columns.values]
    subject_summary = subject_summary.rename(columns={'algorithm_': 'algorithm', 'subject_': 'subject'})
    
    # 调整列顺序，让algorithm列排在subject列前面
    cols = subject_summary.columns.tolist()
    if 'algorithm' in cols and 'subject' in cols:
        cols.remove('algorithm')
        cols.remove('subject')
        cols.insert(1, 'algorithm')
        cols.insert(2, 'subject')
        subject_summary = subject_summary[cols]
    
    return subject_summary


def parse_subjects(subjects_str, dataset='BCI_IV_2A'):
    """
    解析受试者范围字符串，支持多种格式：
    - "all" 表示所有受试者（根据数据集自动确定）
    - "1~9" 表示 1 到 9
    - "1,3,5~7" 表示 1, 3, 5, 6, 7
    - "1 2 3" 表示 1, 2, 3
    """
    subjects = []
    
    # 处理 "all" 关键字
    if subjects_str.lower() == 'all':
        # 根据数据集确定所有受试者的范围
        if dataset == 'BCI_IV_2A':
            subjects = list(range(1, 10))  # 1-9
        elif dataset == 'PhysionetMI':
            subjects = list(range(1, 110))  # 1-109
        elif dataset == 'Schirrmeister2017':
            subjects = list(range(1, 15))  # 1-14
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return subjects
    
    # 处理逗号分隔的字符串
    if ',' in subjects_str:
        parts = subjects_str.split(',')
    else:
        parts = [subjects_str]
    
    for part in parts:
        part = part.strip()
        if '~' in part:
            # 处理范围，如 "1~9"
            start, end = map(int, part.split('~'))
            subjects.extend(range(start, end + 1))
        elif ' ' in part:
            # 处理空格分隔的数字，如 "1 2 3"
            subjects.extend(map(int, part.split()))
        else:
            # 处理单个数字
            subjects.append(int(part))
    
    # 去重并排序
    subjects = sorted(list(set(subjects)))
    return subjects


def main():
    parser = argparse.ArgumentParser(description='BCI IV 2A Motor Imagery Algorithm Evaluation v4 (MOABB)')
    parser.add_argument('--subjects', type=str, default='1~9',
                        help='Subject range or list (e.g., "all", "1~9", "1,3,5~7", or "1 2 3")')
    parser.add_argument('--subject', type=int, default=None,
                        help='Single subject ID - alternative to --subjects')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['CSP+LDA', 'CSP+SVM', 'FBCSP', 'MDM', 'FilterBankTangentSpace+SVM',
                        'RiemannTangentSpace', 'RiemannTangentSpace+SVM', 'RiemannTangentSpace+RF',  'RiemannTangentSpace+PCA', 
                        'EEGNet', 'EEGNex', 'EEG-Inception', 'ShallowFBCSPNet', 
                        'MSVTNet', 'IFNet', 'EEGConformer', 'CTNet', 'ATCNet', 'EEGSimpleConv', 'EEGTCNet',
                         'SincShallowNet', 'EEGITNet'],
                        help='List of algorithms to evaluate')
    parser.add_argument('--use-test-data', action='store_true',
                        help='Use test data (E) instead of training data (T)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and show plots')
    parser.add_argument('--check-gpu', action='store_true',
                        help='Check GPU availability and exit')
    parser.add_argument('--dataset', type=str, default='BCI_IV_2A',
                        choices=['BCI_IV_2A', 'PhysionetMI', 'Schirrmeister2017'], 
                        help='Dataset to use (default: BCI_IV_2A)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained models')
    parser.add_argument('--load-model', action='store_true',
                        help='Load trained models instead of training from scratch')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load models (default: models)')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE feature visualizations')
    
    args = parser.parse_args()
    
    if args.check_gpu:
        check_gpu()
        return
    
    if args.subject is not None:
        args.subjects = [args.subject]
    else:
        # 解析受试者范围
        args.subjects = parse_subjects(args.subjects, dataset=args.dataset)
    
    print("=" * 80)
    print("BCI IV 2A Motor Imagery Algorithm Evaluation System v4 (MOABB)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithms: {', '.join(args.algorithms)}")
    print(f"  Subjects: {', '.join(map(str, args.subjects))}")
    print(f"  Data type: {'Test (E)' if args.use_test_data else 'Training (T)'}")
    print(f"  Training: {args.epochs} epochs")
    print(f"  Data loader: MOABB")
    
    all_results = []
    all_features_dict = {}  # Store features for each subject
    
    for subject_id in args.subjects:
        subject_results = evaluate_subject(subject_id, args.algorithms, use_test_data=args.use_test_data, dataset=args.dataset, epochs=args.epochs, save_model=args.save_model, load_model=args.load_model, model_dir=args.model_dir, extract_features=args.tsne)
        
        if args.tsne:
            results, features_dict = subject_results
            all_results.extend(results)
            all_features_dict[subject_id] = features_dict
        else:
            all_results.extend(subject_results)
    
    print("\n" + "=" * 80)
    print("Evaluation complete, processing results...")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    summary_df = generate_summary(all_results)
    subject_summary_df = generate_subject_summary(all_results)
    
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))
    
    print("\nSubject-wise Performance Summary:")
    print(subject_summary_df.to_string(index=False))
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 保存结果，添加数据集名称和时间戳到文件名
    dataset_suffix = args.dataset.lower().replace('_', '')
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    results_filename = get_timestamped_filename(f'evaluation_results_{dataset_suffix}', 'csv')
    summary_filename = get_timestamped_filename(f'evaluation_summary_{dataset_suffix}', 'csv')
    subject_summary_filename = get_timestamped_filename(f'evaluation_summary_by_subject_{dataset_suffix}', 'csv')
    
    results_df.to_csv(os.path.join(RESULTS_PATH, results_filename), index=False)
    summary_df.to_csv(os.path.join(RESULTS_PATH, summary_filename), index=False)
    subject_summary_df.to_csv(os.path.join(RESULTS_PATH, subject_summary_filename), index=False)
    
    if args.plot:
        visualization.generate_all_plots(all_results, summary_df)
    
    # Generate t-SNE visualizations if requested
    if args.tsne:
        print("\n" + "=" * 80)
        print("Generating t-SNE feature visualizations...")
        print("=" * 80)
        
        for subject_id in args.subjects:
            if subject_id not in all_features_dict:
                continue
                
            features_dict = all_features_dict[subject_id]
            
            # Load data to get labels
            X, y, meta = data_loader.load_single_subject_moabb(subject_id, use_test_data=args.use_test_data, dataset=args.dataset)
            
            print(f"\nGenerating t-SNE visualizations for Subject {subject_id}...")
            
            # Generate individual t-SNE plots for each algorithm
            for algo_name in args.algorithms:
                if algo_name in features_dict:
                    features = features_dict[algo_name]
                    tsne_filename = get_timestamped_filename(f'tsne_{args.dataset.lower()}_subject{subject_id}_{algo_name.replace("+", "_").replace("-", "_")}', 'png')
                    save_path = os.path.join(RESULTS_PATH, tsne_filename)
                    visualization.plot_tsne_visualization(
                        features, y, algo_name, subject_id, args.dataset, save_path=save_path
                    )
            
            # Generate comparison t-SNE plot with all algorithms
            comparison_filename = get_timestamped_filename(f'tsne_comparison_{args.dataset.lower()}_subject{subject_id}', 'png')
            save_path = os.path.join(RESULTS_PATH, comparison_filename)
            visualization.plot_tsne_comparison(
                features_dict, y, args.algorithms, subject_id, args.dataset, save_path=save_path
            )
        
        print("\nt-SNE visualizations complete!")
    
    print("\n" + "=" * 80)
    print("Evaluation task complete!")
    print("=" * 80)
    print("\nResult files:")
    print(f"  Detailed results: {os.path.join(RESULTS_PATH, results_filename)}")
    print(f"  Summary: {os.path.join(RESULTS_PATH, summary_filename)}")
    print(f"  Subject-wise summary: {os.path.join(RESULTS_PATH, subject_summary_filename)}")
    if args.plot:
        print(f"  Plots: {RESULTS_PATH}/ directory")
    if args.tsne:
        print(f"  t-SNE visualizations: {RESULTS_PATH}/ directory")


if __name__ == '__main__':
    main()
