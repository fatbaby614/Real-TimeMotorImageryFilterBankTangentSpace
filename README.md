# Real-Time Motor Imagery BCI System Using Filter Bank Tangent Space

## Project Overview

This project implements a real-time Motor Imagery (MI) Brain-Computer Interface (BCI) system based on the **Filter Bank Tangent Space (FBTS)** algorithm. The system processes Electroencephalography (EEG) signals using Riemannian geometry methods, achieving efficient and lightweight four-class motor imagery recognition for real-time game control, assistive device control, and other applications.

### Key Features

- **Efficient Algorithm**: Based on Riemannian geometry tangent space projection, no need for large training datasets like deep learning
- **Real-time Performance**: Single inference latency below 20ms, supporting real-time control
- **Lightweight**: Runs on standard CPU, no GPU acceleration required
- **Cross-Session Stability**: Supports cross-session validation (Session 1 training, Session 2 testing)
- **Complete Application**: Includes full workflow from data acquisition, model training, real-time control to game applications

---

## Technical Principles

### 1. Filter Bank Decomposition

EEG signals are first decomposed through multiple band-pass filters to extract features from different frequency bands:

- Band 1: 8-10 Hz (mu rhythm)
- Band 2: 10-16 Hz (mu/beta transition)
- Band 3: 16-24 Hz (beta rhythm)
- Band 4: 24-32 Hz (high-frequency beta)

### 2. Covariance Matrix Estimation

Calculate covariance matrices for signals in each frequency band, using the OAS (Oracle Approximating Shrinkage) estimator to improve estimation stability with small samples.

### 3. Tangent Space Projection

Project covariance matrices from the Riemannian manifold to tangent space, converting non-Euclidean data into Euclidean feature vectors:

- Reference Point: Use geometric mean as reference point
- Projection: Logarithmic mapping projects SPD matrices to tangent space
- Vectorization: Extract upper triangular part as features

### 4. Feature Fusion & Classification

Concatenate tangent space features from all frequency bands and classify using SVM:

- Feature Selection: Use Fisher ratio to select most discriminative features
- Classifier: Support Vector Machine (SVM) with RBF kernel

---

## Project Structure

```
MI_realtime_TangentSpace/
├── algorithms/                 # Algorithm implementations
│   └── fbcsp.py               # FBCSP algorithm implementation
├── config/                     # Configuration files
│   ├── mi_config.py           # Motor imagery parameter configuration
│   └── algorithms_config.py   # Algorithm configuration
├── experiments/                # Experiment scripts
│   ├── ablation_study.py      # Ablation study
│   ├── latency_benchmark.py   # Latency benchmark testing
│   └── statistical_analysis.py # Statistical analysis
├── data/                       # Data storage
├── models/                     # Model storage
├── results/                    # Experiment results
├── res/                        # Resource files (images, etc.)
├── paper/                      # Paper-related files
│
├── algorithms_collection.py    # Algorithm collection and model definitions
├── data_loader_moabb.py       # MOABB dataset loader
├── data_acquisition.py        # Real-time data acquisition
├── train_model.py             # Model training
├── calibrate_model.py         # Model calibration
├── evaluate_algorithms.py     # Algorithm evaluation
├── realtime_control.py        # Real-time control main program
├── mi_maze_game.py            # Maze game
├── mi_tetris_game.py          # Tetris game
└── mi_test_itr.py             # ITR testing
```

---

## Installation and Configuration

### Requirements

- Python 3.8+
- Operating System: Windows/Linux/macOS
- Hardware: Standard CPU, no GPU required

### Dependency Installation

```bash
# Create virtual environment
conda create -n mi_bci python=3.10
conda activate mi_bci

# Install core dependencies
pip install numpy scipy scikit-learn matplotlib
pip install mne pyriemann
pip install braindecode torch
pip install pygame pylsl
pip install moabb
pip install pandas seaborn
```

### OpenBCI Configuration

1. Use OpenBCI Cyton board for EEG signal acquisition
2. 8-channel configuration: C3, C4, Cz, F3, F4, Fz, T3, T4
3. Sampling rate: 250 Hz
4. Transmit data via Lab Streaming Layer (LSL)

---

## Usage

### 1. Data Acquisition

Run the data acquisition program to record motor imagery data:

```bash
python data_acquisition.py --subject 1 --session 1
```

Parameters:

- `--subject`: Subject ID
- `--session`: Session ID
- `--trials-per-class`: Number of trials per class (default: 40)

Workflow:

1. Start OpenBCI and ensure LSL stream is running normally
2. Run the data acquisition script
3. Press spacebar to start
4. Follow on-screen prompts to perform motor imagery tasks:
   - ↑ Tongue imagination
   - ↓ Both feet imagination
   - ← Left hand imagination
   - → Right hand imagination

### 2. Model Training

Train the FBTS model using collected data:

```bash
python train_model.py data/subject_1_session_1.mat --algorithm filterbank_tangent
```

Parameters:

- `mat_files`: Training data file path
- `--algorithm`: Algorithm selection (`fbcsp` or `filterbank_tangent`)
- `--output-dir`: Model output directory

### 3. Model Calibration

Quick calibration for new session data:

```bash
python calibrate_model.py models/fbcsp_model data/subject_1_session_2.mat --eval
```

### 4. Real-Time Control

Start the real-time control program:

```bash
python realtime_control.py models/fbcsp_model
```

Control Mapping:

- Tongue imagination → Up
- Both feet imagination → Down
- Left hand imagination → Left
- Right hand imagination → Right

### 5. Game Applications

#### Maze Game

```bash
python mi_maze_game.py models/fbcsp_model
```

Control Method:

- Use motor imagery to control character movement
- Tongue/Both feet/Left hand/Right hand imagination correspond to up/down/left/right movement
- Find the exit to win

#### Tetris

```bash
python mi_tetris_game.py models/fbcsp_model
```

Control Mapping:

- Tongue imagination → Rotate block
- Both feet imagination → Soft drop
- Left hand imagination → Move left
- Right hand imagination → Move right

### 6. ITR Testing

Test Information Transfer Rate:

```bash
python mi_test_itr.py models/fbcsp_model --trials 20
```

---

## Core Code Description

### algorithms_collection.py

Contains implementations of all algorithms:

- `FilterBankTangentSpace`: Filter Bank Tangent Space algorithm (core algorithm)
- `get_algorithm()`: Get instance of specified algorithm
- Supported algorithms: CSP+LDA, FBCSP, MDM, RiemannTangentSpace, EEGNet, EEGTCNet, etc.

### realtime_control.py

Real-time control main program:

- `connect_lsl()`: Connect to LSL data stream
- `load_model()`: Load trained model
- `classify_window()`: Classify sliding window data
- `main_loop()`: Main control loop

### data_acquisition.py

Data acquisition module:

- `collect_mi_data()`: Collect motor imagery data
- `run_trial()`: Run single trial
- `save_mat()`: Save data in MAT format

### evaluate_algorithms.py

Algorithm evaluation script:

- Supports cross-validation for multiple algorithms
- Generates confusion matrices, t-SNE visualizations
- Outputs accuracy, Kappa coefficient, and other metrics

---

## Algorithm Comparison

| Algorithm | Accuracy (BCI IV 2A) | Training Time | Inference Time | GPU Required |
| --------- | -------------------- | ------------- | -------------- | ------------ |
| FBTS+SVM  | 72.85%               | 5.59s         | 17.04ms        | No           |
| EEGNet    | 58.48%               | ~10min        | ~50ms          | Yes          |
| EEGTCNet  | 59.88%               | ~15min        | ~60ms          | Yes          |
| EEGITNet  | 44.98%               | ~12min        | ~55ms          | Yes          |
| IFNet     | 67.09%               | ~20min        | ~70ms          | Yes          |
| FBCSP     | 68.22%               | 8.23s         | 15.23ms        | No           |

*Note: Deep learning algorithms show significant performance degradation in cross-session scenarios, while FBTS remains stable*

---

## Performance Metrics

### BCI IV 2a Dataset

- **Accuracy**: 72.85% (Cross-session: Session 1 training, Session 2 testing)
- **Kappa Coefficient**: 0.638
- **Training Time**: 5.59 seconds (including hyperparameter optimization)
- **Inference Time**: 17.04 milliseconds/trial

### PhysioNet MI Dataset

- **Accuracy**: 51.95% (109 subjects, four-class random probability 25%)
- **Training Time**: 6.47 seconds

### Real-Time Performance

- **Online Accuracy**: 90%
- **ITR**: 20.59 bits/min
- **Latency**: <20ms

---

## Experiment Scripts

### Ablation Study

```bash
python experiments/ablation_study.py
```

Analyze the impact of different components on performance.

### Latency Benchmark

```bash
python experiments/latency_benchmark.py
```

Test end-to-end system latency.

### Statistical Analysis

```bash
python experiments/statistical_analysis.py
```

Generate statistical reports and significance tests.

---

## Notes

1. **Data Quality**: Ensure electrode impedance is below 5kΩ for good signal quality
2. **Subject Training**: New subjects need some training time to master motor imagery techniques
3. **Environment Control**: Avoid electromagnetic interference, maintain quiet environment
4. **Model Saving**: Trained models are saved in the `models/` directory
5. **Results Viewing**: Experiment results are saved in the `results/` directory

---


---

## Acknowledgments

- [MOABB](https://github.com/NeuroTechX/moabb): Mother of All BCI Benchmarks
- [PyRiemann](https://github.com/pyRiemann/pyRiemann): Riemannian Geometry Toolkit
- [MNE-Python](https://mne.tools/): EEG Signal Processing Toolkit
- [Braindecode](https://braindecode.ai/): Deep Learning EEG Decoding Library
