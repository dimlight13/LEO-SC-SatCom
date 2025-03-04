# DRL-SatCom

## Overview
This repository is the official implementation of the paper “Adaptive Semantic-Empowered LEO Satellite Communication System”, which includes the training and evaluation framework for a VQ-VAE-based model using switching batch normalization (SBN), the training of an estimator to mitigate Doppler effects, and the training of a reinforcement learning agent to optimize the transmission processes. The codebase includes pre-training scripts, evaluation tools, and a GUI-based visualization module.

## Code Structure

### 1. Pretraining: `pretrain_sbn_model.py`
- Trains a **VQ-VAE model** with **SBN** for reconstruction tasks.
- The model is trained on an **AWGN-fixed channel**, which results in significant performance degradation when applied to **Rician** or **Rayleigh** fading channels.
- During training, sample reconstructions are saved every **10 epochs**.
- The pretraining process runs for **400 epochs** in total.
  
- **Key training consideration:**  
  - During the training iterations, the **recovered indices must not be used**; instead, training should directly utilize the **quantized vectors**.

### 2. Evaluation: `eval_ber_psnr.py`
- **BER Evaluation:**  
  - Computes **Bit Error Rate (BER)** performance across different **channel** and **Doppler** conditions **without** loading the pretrained model.
- **PSNR Evaluation:**  
  - Computes **Peak Signal-to-Noise Ratio (PSNR)** performance under different conditions **with** the pretrained model.
- Supports **both coded and uncoded** channel configurations.

### 3. GUI Application: `app_tk.py`
#### Features & Configuration Options  

- **UI Execution**:  
  - Launching the script opens a **graphical user interface**, allowing users to adjust **parameters and visualize performance metrics**.  
  - The **PSNR and BER buttons** allow users to switch between performance metrics dynamically.  

- **Doppler Type**:  
  - `multi`: Simulates a **multiple Doppler environment**.  
  - `none`: Assumes a **general channel condition without Doppler effects**.  

- **Channel Type**:  
  - Options: `'awgn'`, `'rayleigh'`, `'rician'`.  
  - **Doppler is fixed to `'awgn'`**, meaning that changing the **Doppler Type** will automatically set **Channel Type** to `'awgn'`.  

- **Modulation Scheme**:  
  - Supported schemes: **`BPSK`, `QPSK`, `16QAM`, `64QAM`, `256QAM`, `auto`**.  
  - Selecting `auto` enables **automatic modulation adaptation** based on a predefined **SNR range**.  

- **Channel Coding**:  
  - Options: `'both'`, `True`, `False`.  
  - `'both'`: **Plots results for both coded and uncoded scenarios** (may slow down performance).  
  - `True`: **Displays results only for channel-coded transmission**.  
  - `False`: **Displays results only for uncoded transmission**.  

- **SNR Configuration**:  
  - Users can manually adjust **minimum and maximum SNR values (in dB)**.  

- **Execution**:  
  - After configuring all parameters, clicking **"Run Simulation"** will generate and display the **performance plots** automatically.  

## Getting Started

### Installation
```sh
pip install -r requirements.txt
```

### Usage

#### Pretraining the Model
```sh
python pretrain_sbn_model.py --epochs 400 --save_interval 10
```

#### Evaluating BER and PSNR
```sh
python eval_ber_psnr.py --eval_type ber --snr_min -10 --snr_max 30  # For BER evaluation
python eval_ber_psnr.py --eval_type psnr --snr_min -10 --snr_max 30 # For PSNR evaluation
```

#### Running the GUI
```sh
python app_tk.py
```

## Notes
- The **pretrained model** is optimized for an **AWGN** channel. Performance in **fading channels (e.g., Rician, Rayleigh)** will be significantly lower.
- Ensure **modulation index handling** is correctly implemented to prevent memory leaks.

## License
This project is licensed under the MIT License
