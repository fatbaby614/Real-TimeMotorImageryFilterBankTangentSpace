from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from algorithms.fbcsp import FilterBankCSPClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an existing FBCSP model with calibration data")
    parser.add_argument("model_dir", help="Directory containing the pretrained FBCSP model")
    parser.add_argument("mat_files", nargs="+", help="Calibration MAT files (small new dataset)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination for the calibrated model (defaults to <model_dir>_calibrated_<timestamp>)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Report calibration accuracy on the provided MAT data",
    )
    return parser.parse_args()


def load_trials(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    data, labels = [], []
    for path in paths:
        mat = loadmat(path, squeeze_me=True)
        data.append(np.asarray(mat["data"], dtype=np.float32))
        labels.append(np.asarray(mat["labels"], dtype=np.int32))
    X = np.concatenate(data, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def main() -> None:
    args = parse_args()
    base_dir = Path(args.model_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {base_dir}")
    model = FilterBankCSPClassifier.load(base_dir)
    X_cal, y_cal = load_trials(args.mat_files)
    if X_cal.ndim != 3:
        raise ValueError("Calibration data must be shaped (trials, channels, samples)")

    # Reuse the loaded filter bank + spatial filters, only refit scaler + SVM.
    raw_features = model._extract_features(X_cal, fit_csp=False)  # type: ignore[attr-defined]
    scaled_features = model._scaler.fit_transform(raw_features)
    model._svm.fit(scaled_features, y_cal)

    metrics = {
        "calibration_files": args.mat_files,
        "calibration_trials": int(X_cal.shape[0]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if args.eval:
        preds = model._svm.predict(scaled_features)
        metrics["calibration_accuracy"] = float(accuracy_score(y_cal, preds))
        print(f"Calibration accuracy: {metrics['calibration_accuracy']:.3f}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir.parent / f"{base_dir.name}_calibrated_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    (out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Calibrated model saved to {out_dir}")


if __name__ == "__main__":
    main()
