# pix2pix-CGANs

This repository provides an implementation of **Pix2Pix / Conditional Generative Adversarial Networks (cGANs)** for *paired image-to-image translation* tasks. The code is mainly intended for **research and experimental purposes**, and is suitable for applications such as:

* Remote sensing image → target map (e.g., DEM, masks, indices)
* Grayscale image → color image
* Sketch → realistic image

The implementation is based on **TensorFlow 1.x + Keras**, following the classic Pix2Pix framework.

---

## 1. Environment Setup

### 1.1 Operating System

* Windows 10 / 11 (tested)
* Linux (Ubuntu 18.04+ should also work)

### 1.2 Python Version

* **Python 3.6 – 3.7** (recommended)

> ⚠ This project relies on **TensorFlow 1.13**, which is **not compatible with Python ≥ 3.8**.

---

### 1.3 Deep Learning Framework

* TensorFlow-GPU **1.13.1**
* TensorFlow-Estimator **1.13.0**
* Keras **2.1.5**

GPU training requires:

* CUDA 10.0
* cuDNN 7.x

---

### 1.4 Python Dependencies

All required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

Key packages include:

```text
tensorflow-gpu==1.13.1
keras==2.1.5
numpy==1.17.0
opencv-python==4.1.2.30
matplotlib==3.1.2
scikit-image==0.17.2
imageio==2.15.0
tqdm==4.60.0
```

> ⚠ It is strongly recommended to use a **virtual environment or Conda environment** to avoid dependency conflicts.

---

## 2. Project Structure

A typical directory structure is shown below:

```text
pix2pix-CGANs/
│
├── train_input/        # Training input images (Domain A)
├── train_target/       # Training target images (Domain B)
├── test_input/         # Test input images
├── test_target/        # Test target images (optional)
│
├── pix2pix_LMQ (1).py  # Main script (model + training + inference)
├── requirements.txt   # Environment configuration
└── README.md
```

---

## 3. Data Preparation

Pix2Pix is a **supervised GAN**, therefore **paired data are mandatory**.

### Requirements:

* Each input image must have a corresponding target image
* File names and ordering must be strictly consistent
* Input and target images must have the same spatial resolution

Example:

```text
train_input/  image_001.png
```

Recommended image size:

* 256 × 256 (default Pix2Pix setting)
* 512 × 512 also supported if GPU memory allows

---

## 4. Method Overview

The model follows the standard **Pix2Pix (cGAN)** architecture:

* **Generator**: U-Net–based encoder–decoder network
* **Discriminator**: PatchGAN classifier
* **Loss function**:

  * Conditional GAN loss
  * L1 reconstruction loss

Overall objective:

```
L = L_cGAN + λ · L_L1
```

This design encourages both **structural realism** and **pixel-level fidelity**.

---

## 5. Usage

### 5.1 Training

After preparing the dataset, run the main script:

```bash
python pix2pix_LMQ\ \(1\).py
```

Training parameters such as:

* number of epochs
* batch size
* learning rate
* input image size

can be modified directly inside the script.

---

### 5.2 Outputs

During and after training:

* Model checkpoints are saved to the specified directory (e.g., `checkpoints/`)
* Generated images are saved to an output directory (e.g., `train_output/`)

---

### 5.3 Inference / Testing

Place unseen images into:

```text
test_input/
```

Run the script in inference mode (if implemented) to generate corresponding outputs.

---

## 6. Common Issues

### TensorFlow version conflict

Ensure that TensorFlow 1.13 is used. TensorFlow 2.x is **not supported** by this codebase.

### CUDA compatibility

TensorFlow 1.13 requires CUDA 10.0. Using newer CUDA versions may cause runtime errors.

---

## 7. References

* Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR.
* Official Pix2Pix implementations (TensorFlow / PyTorch)

---

## 8. Disclaimer

This code is provided for **academic research and educational purposes only**. Users are responsible for verifying data quality and model reliability before applying it to real-world tasks.

---

## 9. Author

* Author: AlanNUIST
* Affiliation: Nanjing University of Information Science & Technology (NUIST)
* Research Interests: Remote sensing, DEM modeling, terrain reconstruction with deep learning

---

Feel free to improve this repository by adding documentation or submitting pull requests.
