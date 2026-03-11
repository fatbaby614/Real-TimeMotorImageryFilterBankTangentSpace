import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from config.algorithms_config import RESULTS_PATH, get_timestamped_filename
import os
import platform


system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_accuracy_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='algorithm', y='accuracy', data=df, palette='Set2')
    ax = sns.stripplot(x='algorithm', y='accuracy', data=df, color='black', alpha=0.5, size=6)
    
    plt.title('Algorithm Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_kappa_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='algorithm', y='kappa', data=df, palette='Set3')
    ax = sns.stripplot(x='algorithm', y='kappa', data=df, color='black', alpha=0.5, size=6)
    
    plt.title("Cohen's Kappa Comparison", fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Kappa', fontsize=14)
    plt.ylim([0, 1])
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_time(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='algorithm', y='train_time', data=df, palette='viridis', errorbar='sd')
    
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_subject_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(x='subject', y='accuracy', hue='algorithm', data=df, palette='Set2')
    
    plt.title('Accuracy Comparison by Subject', fontsize=16, fontweight='bold')
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix(results, algorithm_name, save_path=None):
    df = pd.DataFrame(results)
    algorithm_results = df[df['algorithm'] == algorithm_name]
    
    all_cms = []
    for cm_str in algorithm_results['confusion_matrix']:
        cm = np.array(cm_str)
        all_cms.append(cm)
    
    mean_cm = np.mean(all_cms, axis=0)
    
    # 根据混淆矩阵的形状动态确定类别数量
    n_classes = mean_cm.shape[0]
    
    # 根据数据集类型和类别数量动态生成类别标签
    dataset_name = algorithm_results['dataset'].iloc[0] if 'dataset' in algorithm_results.columns else 'Unknown'
    
    if n_classes == 2:
        classes = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            classes = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            classes = ['Left', 'Right', 'Foot', 'Tongue']
    else:
        classes = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'{algorithm_name} Confusion Matrix (Average)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_performance_summary(summary_df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].bar(summary_df['algorithm'], summary_df['accuracy_mean'], 
                   yerr=summary_df['accuracy_std'], capsize=5, color='steelblue')
    axes[0, 0].set_title('Mean Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].bar(summary_df['algorithm'], summary_df['kappa_mean'], 
                   yerr=summary_df['kappa_std'], capsize=5, color='coral')
    axes[0, 1].set_title('Mean Kappa', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Kappa', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].bar(summary_df['algorithm'], summary_df['train_time_mean'], 
                   yerr=summary_df['train_time_std'], capsize=5, color='forestgreen')
    axes[1, 0].set_title('Mean Training Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Time (s)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    axes[1, 1].bar(summary_df['algorithm'], summary_df['accuracy_max'], 
                   alpha=0.7, color='purple', label='Max')
    axes[1, 1].bar(summary_df['algorithm'], summary_df['accuracy_min'], 
                   alpha=0.7, color='orange', label='Min')
    axes[1, 1].set_title('Accuracy Range', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_tsne_visualization(features, labels, algorithm_name, subject_id, dataset_name, save_path=None, use_pca=True):
    """
    Generate t-SNE visualization for features extracted by an algorithm
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    labels : np.ndarray
        True labels of shape (n_samples,)
    algorithm_name : str
        Name of the algorithm
    subject_id : int
        Subject ID
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    use_pca : bool, optional
        Whether to use PCA preprocessing before t-SNE (default: True)
    """
    if features.shape[0] < 3:
        print(f"Warning: Not enough samples ({features.shape[0]}) for t-SNE visualization")
        return
    
    n_classes = len(np.unique(labels))
    
    # Generate class labels based on dataset and number of classes
    if n_classes == 2:
        class_names = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            class_names = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            class_names = ['Left', 'Right', 'Foot', 'Tongue']
    else:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    print(f"  Applying t-SNE on {features.shape} features...")
    
    # Method 2: PCA preprocessing for better t-SNE results
    if use_pca and features.shape[1] > 50:
        print(f"  Applying PCA preprocessing...")
        from sklearn.decomposition import PCA
        # Reduce to 50 dimensions first
        n_pca_components = min(50, features.shape[0] - 1, features.shape[1])
        pca = PCA(n_components=n_pca_components, random_state=42)
        features = pca.fit_transform(features)
        print(f"  PCA reduced dimensions to {features.shape[1]} (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    
    # Method 3: Optimized t-SNE parameters
    n_samples = features.shape[0]
    
    # Better perplexity calculation
    perplexity = min(50, max(5, n_samples // 4))
    
    # Adjust learning rate based on perplexity
    learning_rate = max(100, perplexity * 4)
    
    print(f"  t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}")
    
    # Apply t-SNE with optimized parameters
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity, 
        learning_rate=learning_rate,
        max_iter=2000,
        early_exaggeration=12.0,
        init='pca',
        method='barnes_hut' if n_samples > 1000 else 'exact'
    )
    features_2d = tsne.fit_transform(features)
    
    # Create color palette
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each class separately for better legend
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=class_name, alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    
    pca_text = " (with PCA)" if use_pca and features.shape[1] > 50 else ""
    ax.set_title(f't-SNE Visualization{pca_text}: {algorithm_name}\n{dataset_name} - Subject {subject_id}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=14)
    ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.legend(title='Class', fontsize=12, title_fontsize=13, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  t-SNE plot saved to {save_path}")
    
    plt.close()


def plot_tsne_comparison(features_dict, labels, algorithm_names, subject_id, dataset_name, save_path=None, use_pca=True):
    """
    Generate t-SNE visualization comparing multiple algorithms
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary mapping algorithm names to feature matrices
    labels : np.ndarray
        True labels of shape (n_samples,)
    algorithm_names : list
        List of algorithm names to visualize
    subject_id : int
        Subject ID
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    use_pca : bool, optional
        Whether to use PCA preprocessing before t-SNE (default: True)
    """
    n_algorithms = len(algorithm_names)
    n_cols = min(3, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_algorithms == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    n_classes = len(np.unique(labels))
    
    # Generate class labels based on dataset and number of classes
    if n_classes == 2:
        class_names = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            class_names = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            class_names = ['Left', 'Right', 'Foot', 'Tongue']
    else:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for idx, algo_name in enumerate(algorithm_names):
        ax = axes[idx]
        
        if algo_name not in features_dict:
            ax.text(0.5, 0.5, 'No features available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            continue
        
        features = features_dict[algo_name]
        
        if features.shape[0] < 3:
            ax.text(0.5, 0.5, f'Insufficient samples\n({features.shape[0]})', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            continue
        
        # Apply PCA preprocessing if enabled
        if use_pca and features.shape[1] > 50:
            from sklearn.decomposition import PCA
            n_pca_components = min(50, features.shape[0] - 1, features.shape[1])
            pca = PCA(n_components=n_pca_components, random_state=42)
            features = pca.fit_transform(features)
        
        # Apply t-SNE with optimized parameters
        n_samples = features.shape[0]
        perplexity = min(50, max(5, n_samples // 4))
        learning_rate = max(100, perplexity * 4)
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            learning_rate=learning_rate,
            max_iter=2000,
            early_exaggeration=12.0,
            init='pca',
            method='barnes_hut' if n_samples > 1000 else 'exact'
        )
        features_2d = tsne.fit_transform(features)
        
        # Plot each class separately
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[colors[i]], label=class_name, alpha=0.7, s=50, edgecolors='k', linewidth=0.3)
        
        ax.set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(title='Class', fontsize=9, title_fontsize=10, loc='best')
    
    # Hide unused subplots
    for idx in range(n_algorithms, len(axes)):
        axes[idx].axis('off')
    
    pca_text = " (with PCA)" if use_pca else ""
    plt.suptitle(f't-SNE Feature Comparison{pca_text}\n{dataset_name} - Subject {subject_id}', 
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  t-SNE comparison plot saved to {save_path}")
    
    plt.close()


def generate_all_plots(results, summary_df):
    print("\nGenerating visualizations...")
    
    # Generate timestamped filenames for all plots
    plot_accuracy_comparison(results, save_path=get_timestamped_filename('accuracy_comparison', 'png'))
    plot_kappa_comparison(results, save_path=get_timestamped_filename('kappa_comparison', 'png'))
    plot_training_time(results, save_path=get_timestamped_filename('training_time', 'png'))
    plot_performance_summary(summary_df, save_path=get_timestamped_filename('performance_summary', 'png'))
    
    algorithms = pd.DataFrame(results)['algorithm'].unique()
    for algo in algorithms:
        safe_algo_name = algo.replace('+', '_').replace('-', '_')
        plot_confusion_matrix(results, algo, save_path=get_timestamped_filename(f'confusion_matrix_{safe_algo_name}', 'png'))
    
    print("All plots saved to results/ directory")
