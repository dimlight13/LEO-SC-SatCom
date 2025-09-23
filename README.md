# DRL-SatCom

## Overview
This repository is the official implementation of the paper “Doppler-Adaptive Digital Semantic Communication for Low Earth Orbit Satellite Systems”, which includes the training and evaluation framework for a C-VQ-VAE-based model using switching batch normalization (SBN), the training of an estimator to mitigate Doppler effects, and the training of a reinforcement learning agent to optimize the transmission processes. The codebase includes pre-training scripts, evaluation tools, and a GUI-based visualization module.

## Code Structure

### 1. Pretraining: `train_c_vq_vae_model.py`
- Trains a **C-VQ-VAE model** for digital-compatible reconstruction tasks.
- The model is trained on an **AWGN-fixed channel**, which results in significant performance degradation when applied to **Rician** or **Rayleigh** fading channels.
- The total number of training epochs is set to be **100**.
  
- **Key training consideration:**  
  - During the training iterations, the **recovered indices must not be used**; instead, training should directly utilize the **quantized vectors**.

### 2. Save PSNR and symbol data for efficient training of PPO agent and post equalizer: `save_psnr_data.py` and `save_symbol_data.py`
- Executes two saving codes for efficient training of the PPO agent and the post-equalizer.
- This is a tricky implementation technique, as running without stored code would take an extremely long time to train.
- Data is stored inside `doppler_data` folder.

### 3. Training: `train_tx_agent_ppo.py`
- Trains a **PPO agent** applied to the transmitter for adaptive modulation selection.
- The total number of training epochs is set to be **200**.

### 4. Training: `train_post_model.py`
- Trains a **post-equalizer** applied to the receiver for mitigation of residual Doppler effects.
- The total number of training epochs is set to be **50**.

#### Running the GUI
```sh
python app_tk.py
```
- **UI Execution**:  
  - Launching the script opens a **graphical user interface**, allowing users to adjust **parameters and visualize performance metrics**.  

### Parameter Configuration Guide

**Dataset Options:**
- `cifar10`: 32x32 RGB images, 10 classes
- `eurosat`: 64x64 RGB satellite images, 10 classes

**Modulation Types:**
- `SC_auto`: Smart coding with automatic modulation
- `SC_none`: Smart coding without modulation optimization
- `TN_auto`: Traditional method with automatic modulation  
- `BPSK`, `QPSK`, `16QAM`, `64QAM`, `256QAM`: Specific modulation schemes

**Channel Coding:**
- `true`: Enable channel coding (LDPC codes)
- `false`: Disable channel coding
- `both`: Compare both scenarios

**Compensation Methods:**
- `lmmse`: Linear Minimum Mean Square Error
- `mrc`: Maximum Ratio Combining
- `none`: No compensation

- **SNR Configuration**:  
  - Users can manually adjust **minimum and maximum SNR values (in dB)**.  

- **Execution**:  
  - After configuring all parameters, clicking **"Run Simulation"** will generate and display the **performance plots** automatically.  

## License
This project is licensed under the MIT License
