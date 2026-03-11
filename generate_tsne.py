"""Generate t-SNE visualization for FilterBankTangentSpace model."""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat

from algorithms_collection import FilterBankTangentSpace
from visualization import plot_tsne_visualization
from config import mi_config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization for FilterBankTangentSpace model")
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("mat_files", nargs="+", help="Paths to MAT dataset files for visualization")
    parser.add_argument("--save-dir", default="results", help="Directory to save t-SNE plot")
    parser.add_argument("--subject-id", type=int, default=1, help="Subject ID for plot title")
    parser.add_argument("--dataset-name", default="OpenBCI", help="Dataset name for plot title")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA preprocessing before t-SNE")
    return parser.parse_args()


def load_datasets(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load datasets from MAT files."""
    trials, labels = [], []
    for path in paths:
        mat = loadmat(path, squeeze_me=True)
        trials.append(np.asarray(mat["data"], dtype=np.float32))
        labels.append(np.asarray(mat["labels"], dtype=np.int32))
    X = np.concatenate(trials, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def main() -> None:
    args = parse_args()
    
    # Load the model
    model_path = Path(args.model_dir) / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = FilterBankTangentSpace.load_model(model_path)
    print("Model loaded successfully!")
    
    # Load datasets
    print(f"Loading data from {len(args.mat_files)} files...")
    X, y = load_datasets(args.mat_files)
    print(f"Loaded {X.shape[0]} trials with shape {X.shape[1:]} each")
    
    # Extract features
    print("Extracting features...")
    features = model.extract_features(X)
    print(f"Extracted features with shape: {features.shape}")
    
    # Generate t-SNE visualization
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "FilterBankTangentSpace"
    save_path = save_dir / f"tsne_{model_name}_subject{args.subject_id}_{Path(args.mat_files[0]).stem}_{Path(args.model_dir).stem}.png"
    
    print(f"Generating t-SNE visualization...")
    plot_tsne_visualization(
        features=features,
        labels=y,
        algorithm_name=model_name,
        subject_id=args.subject_id,
        dataset_name=args.dataset_name,
        save_path=str(save_path),
        use_pca=args.use_pca
    )
    
    print(f"t-SNE plot saved to {save_path}")


if __name__ == "__main__":
    main()
