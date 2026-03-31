# Real-Time Motor Imagery BCI Testing Guide

This guide provides step-by-step instructions for setting up and running the real-time motor imagery BCI system using OpenBCI hardware.

## Hardware Requirements

- **OpenBCI Cyton Board** (8-channel EEG acquisition system)
- **Electrode Cap**: Choose one of:
  - OpenBCI Gel Cap (8-channel)
  - OpenBCI Ultracortex Dry Cap (8-channel)
- **Computer** with Bluetooth capability
- **USB Dongle** (for OpenBCI Cyton Bluetooth connection)

## Software Requirements

- **Python 3.8+**
- **Required Python packages**:
  ```
  pip install numpy scipy matplotlib scikit-learn pyriemann mne pyqt5 pyserial
  ```
- **OpenBCI GUI** (for initial hardware setup)
- **Lab Streaming Layer (LSL)**

## Setup Instructions

### 1. Hardware Setup

1. **Prepare the electrode cap**:
   - For gel cap: Apply conductive gel to each electrode
   - For dry cap: Ensure electrodes make good contact with the scalp

2. **Connect OpenBCI Cyton**:
   - Turn on the Cyton board
   - Pair via Bluetooth using the OpenBCI GUI or direct Bluetooth connection

3. **Verify signal quality**:
   - Check EEG signals in the OpenBCI GUI
   - Ensure all channels have good signal quality (minimal noise)

### 2. Software Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fatbaby614/Real-TimeMotorImageryFilterBankTangentSpace.git
   cd Real-TimeMotorImageryFilterBankTangentSpace
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not pre-trained):
   ```bash
   python train_model.py --method filterbank_tangent --dataset bci_iv_2a
   ```

## Running Real-Time Tests

### Option 1: ITR (Information Transfer Rate) Test

This test measures the system's performance in a controlled setting.

1. **Start the ITR test**:
   ```bash
   python mi_test_itr.py
   ```

2. **Follow the on-screen instructions**:
   - The system will display cues for different motor imagery tasks
   - Perform the specified motor imagery for each trial
   - After 60 trials, the system will calculate and display ITR results

### Option 2: Tetris Game Control

Control a Tetris game using motor imagery commands.

1. **Start the Tetris game**:
   ```bash
   python mi_tetris_game.py
   ```

2. **Control scheme**:
   - Left hand imagery: Move piece left
   - Right hand imagery: Move piece right
   - Feet imagery: Rotate piece
   - Tongue imagery: Drop piece

### Option 3: Maze Navigation Game

Navigate a character through a maze using motor imagery.

1. **Start the maze game**:
   ```bash
   python mi_maze_game.py
   ```

2. **Control scheme**:
   - Left hand imagery: Move left
   - Right hand imagery: Move right
   - Feet imagery: Move forward
   - Tongue imagery: Restart level

## Data Acquisition

To record EEG data during real-time sessions:

```bash
python data_acquisition.py --output_dir data/ --session_name test_session
```

## Troubleshooting

### Common Issues

1. **Bluetooth connection problems**:
   - Reset the Cyton board
   - Re-pair the device
   - Check USB dongle connection

2. **Poor signal quality**:
   - For gel cap: Apply more conductive gel
   - For dry cap: Adjust electrode tension
   - Ensure proper electrode placement

3. **Low classification accuracy**:
   - Retrain the model with more data
   - Check electrode contact quality
   - Ensure consistent motor imagery practice

## System Architecture

The real-time BCI system consists of five layers:

1. **Hardware Layer**: OpenBCI Cyton board and electrode cap
2. **Data Acquisition Layer**: LSL stream inlet and signal preprocessing
3. **Signal Processing Layer**: Filter bank decomposition and tangent space mapping
4. **Machine Learning Layer**: SVM classification and model inference
5. **Application Layer**: Game control interfaces

## Performance Metrics

- **Training time**: ~5 seconds
- **Inference latency**: <20 ms per trial
- **Update rate**: 50 Hz (20 ms intervals)

## Citation

If you use this system in your research, please cite:



## Contact

For questions or issues, please contact:
- Huang Tan: fatbaby@163.com
- GitHub Issues: https://github.com/fatbaby614/Real-TimeMotorImageryFilterBankTangentSpace/issues