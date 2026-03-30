# 🔍 ReID ETLT Converter

> **NVIDIA ReID Model Conversion Pipeline** — Convert a `.etlt` encrypted model into a **TensorRT engine** for fast Person Re-Identification on your desktop GPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x%2B-orange.svg)](https://developer.nvidia.com/tensorrt)
[![TAO Toolkit](https://img.shields.io/badge/TAO%20Toolkit-5.2.0-purple.svg)](https://developer.nvidia.com/tao-toolkit)

---

## 📖 Table of Contents

- [What Is This?](#-what-is-this)
- [Understanding the Technologies](#-understanding-the-technologies)
  - [Person Re-Identification (ReID)](#1-person-re-identification-reid)
  - [ResNet-50 — The Neural Network Backbone](#2-resnet-50--the-neural-network-backbone)
  - [NVIDIA TAO Toolkit](#3-nvidia-tao-toolkit)
  - [The .etlt File Format](#4-the-etlt-file-format)
  - [ONNX — Open Neural Network Exchange](#5-onnx--open-neural-network-exchange)
  - [TensorRT — The Inference Engine](#6-tensorrt--the-inference-engine)
  - [FP16 — Half Precision Inference](#7-fp16--half-precision-inference)
  - [Docker and NVIDIA Container Toolkit](#8-docker-and-nvidia-container-toolkit)
  - [BNNeck — The Feature Neck](#9-bnneck--the-feature-neck)
- [Pipeline Overview](#-pipeline-overview)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Getting Started](#-getting-started)
- [Conversion Steps](#-conversion-steps)
- [Model Configuration](#️-model-configuration)
- [Frequently Asked Questions](#-frequently-asked-questions)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 What Is This?

This repository provides a **ready-to-use, two-step pipeline** to convert NVIDIA's pre-trained ReID model from its encrypted `.etlt` format into a **TensorRT `.engine` file** that runs blazingly fast on a desktop GPU — **no data center GPU required**.

The final `.engine` file can be plugged into any person tracking or multi-camera surveillance system to perform real-time Person Re-Identification.

---

## 🧠 Understanding the Technologies

Before running anything, it helps to understand *what* each piece of technology is and *why* it exists in this pipeline. This section explains every key concept from the ground up — perfect if you are a student encountering these tools for the first time.

---

### 1. Person Re-Identification (ReID)

#### What is it?

**Person Re-Identification (ReID)** is a computer vision task where a model answers the question:

> *"Is the person I see in Camera A the same person I saw in Camera B?"*

This is different from face recognition. ReID works even when:
- The person's face is not visible
- The camera angle is completely different
- Lighting conditions have changed
- Some time has passed between sightings

#### How does it work?

A ReID model takes an image of a person (called a **crop**) and converts it into a **feature vector** — a fixed-length list of numbers (e.g., 256 numbers) that uniquely describes that person's appearance.

```
Person Image (128×256 pixels)
         │
         ▼
   [ ReID Neural Network ]
         │
         ▼
Feature Vector: [0.12, -0.85, 0.43, 0.71, ..., 0.33]  ← 256 numbers
```

Two images of the **same person** produce feature vectors that are **very close** to each other in mathematical space. Two images of **different people** produce vectors that are **far apart**.

To compare two people, you compute the **cosine similarity** or **Euclidean distance** between their vectors. If the distance is below a threshold → same person. Above the threshold → different person.

#### Where is ReID used?

- Multi-camera surveillance and security systems
- Real-time person tracking in video analytics
- Retail customer journey and footfall analysis
- Smart city and traffic monitoring
- Sports analytics — tracking players across broadcast angles

---

### 2. ResNet-50 — The Neural Network Backbone

#### What is a "backbone"?

A **backbone** is the main feature extraction network inside a deep learning model. Think of it as the engine of a car — it does the heavy lifting of converting a raw image into meaningful numerical features.

#### What is ResNet-50?

**ResNet-50** (Residual Network with 50 layers) is one of the most popular and proven deep learning architectures, introduced by Microsoft Research in 2015. It won the ImageNet Large Scale Visual Recognition Challenge with a top-5 error rate of just 3.57%.

```
Input Image (3 × 256 × 128)
       │
       ▼
  Conv Layer 1       (64 filters, 7×7 kernel)
       │
       ▼
  Residual Block × 3   (64 filters)
       │
       ▼
  Residual Block × 4   (128 filters)
       │
       ▼
  Residual Block × 6   (256 filters)
       │
       ▼
  Residual Block × 3   (512 filters)
       │
       ▼
  Global Average Pooling
       │
       ▼
  2048-dim feature → BNNeck → 256-dim output vector
```

#### What makes ResNet special? The Residual (Skip) Connection

The key innovation of ResNet is the **skip connection** — it adds the input of a block directly to its output:

```
     Input (x)
       │  ╲
       │   ╲ shortcut (identity)
       ▼    ╲
  [ Conv → BatchNorm → ReLU ]
       │    ╱
       ▼   ╱
  [ Conv → BatchNorm ]
       │  ╱
       + ←── adds original input back here
       │
       ▼
     ReLU
     Output = F(x) + x
```

**Why does this matter?** Without skip connections, very deep networks suffer from the **vanishing gradient problem** — as gradients flow back through dozens of layers during training, they shrink to near zero and early layers stop learning. Skip connections give gradients a shortcut path to flow through directly, making training of very deep networks stable and effective.

#### Why ResNet-50 for ReID?

- Deep enough (50 layers) to learn complex appearance features
- Proven accuracy on major ReID benchmarks (Market-1501, DukeMTMC)
- Fast enough for real-time inference with TensorRT
- Well-supported by all major frameworks and TensorRT optimization

---

### 3. NVIDIA TAO Toolkit

#### What is TAO?

**TAO (Train, Adapt, and Optimize)** Toolkit is NVIDIA's end-to-end AI framework for training and deploying computer vision models. It lets you:
- Use NVIDIA's high-quality pre-trained models without training from scratch
- Fine-tune models on your own dataset using just config files
- Export models to deployment formats like ONNX and TensorRT

Think of TAO as a complete AI workshop — it handles data loading, training, evaluation, and export, all through YAML configuration.

#### Why do we need TAO here?

In this pipeline, TAO is used **only for one thing** — decrypting the `.etlt` file and exporting it to `.onnx` format. The command is:

```bash
tao model re_identification export
```

This is a **one-time operation**. Once you have the `.onnx` file, you never need TAO again.

#### TAO Toolkit Docker Image

TAO is distributed exclusively as a Docker container:

```
nvcr.io/nvidia/tao/tao-toolkit:5.2.0-tf2.11.0
```

| Part | Meaning |
|---|---|
| `nvcr.io` | NVIDIA's NGC (GPU Cloud) container registry |
| `nvidia/tao/tao-toolkit` | The TAO Toolkit image |
| `5.2.0` | TAO version |
| `tf2.11.0` | Built on TensorFlow 2.11 |

The image is ~5GB because it bundles TensorFlow, CUDA libraries, cuDNN, and all TAO tools inside a self-contained environment.

---

### 4. The `.etlt` File Format

#### What is `.etlt`?

`.etlt` stands for **Encrypted TLT** — TLT was the old name for TAO Toolkit (Transfer Learning Toolkit). It is NVIDIA's proprietary encrypted model format.

When NVIDIA distributes pre-trained models, they encrypt the weights to:
1. Protect their intellectual property
2. Prevent unauthorized modification
3. Ensure models are only used via the official TAO pipeline

#### How does the encryption work?

```
Original Model Weights (plain .tlt or .pth)
         │
         ▼
  AES Symmetric Encryption  (with a secret key)
         │
         ▼
  .etlt file  ← What NVIDIA distributes
```

To use the model, you need the **decryption key**. For this repo's model, NVIDIA published the key (`nvidia_tao`) openly because it is a community pre-trained model.

#### The model in this repo

| Property | Value |
|---|---|
| File | `resnet50_market1501_aicity156.etlt` |
| Encryption Key | `nvidia_tao` |
| Backbone | ResNet-50 |
| Training Datasets | Market-1501 + AICityChallenge |

**Market-1501** is one of the most widely used ReID benchmarks — 32,668 images of 1,501 person identities captured by 6 cameras at a university marketplace in Beijing.

---

### 5. ONNX — Open Neural Network Exchange

#### What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open-source, standardized format for representing machine learning models. Think of it as a **universal file format for AI models** — like PDF is for documents, ONNX is for neural networks.

```
PyTorch Model  ──┐
TensorFlow Model─┼──► ONNX Format ──► TensorRT (NVIDIA)
Keras Model  ────┘               ──► ONNX Runtime (CPU/GPU)
                                 ──► OpenVINO (Intel)
                                 ──► CoreML (Apple)
```

#### Why convert to ONNX first, not directly to TensorRT?

Directly converting `.etlt` → TensorRT in one step is not supported by TAO's export pipeline. ONNX acts as an essential bridge:

1. **Open and inspectable**: You can open `.onnx` files with tools like [Netron](https://netron.app/) to visualize the full model graph — every layer, every connection
2. **Portable**: ONNX runs on any hardware — NVIDIA, Intel, AMD, ARM
3. **Validated**: You can verify the model works correctly before spending 10+ minutes on TensorRT compilation
4. **Flexible fallback**: If TensorRT isn't available, run inference directly with `onnxruntime-gpu`

#### What does an ONNX file contain?

An ONNX file is a binary file (Protocol Buffer format) containing:
- **Graph**: The full computational graph — how layers connect to each other
- **Nodes**: Every operation (Conv2D, BatchNorm, ReLU, GlobalAvgPool, etc.)
- **Initializers**: The trained weight tensors (all the learned numbers)
- **Input/Output specs**: Tensor names, shapes, and data types

---

### 6. TensorRT — The Inference Engine

#### What is TensorRT?

**NVIDIA TensorRT** is a high-performance deep learning **inference optimizer and runtime**. It takes a trained model and compiles it into a highly optimized **engine** specifically tuned for your exact GPU model.

The analogy: running a neural network from an ONNX file is like running Python code in an interpreter — it works but it's not maximally fast. TensorRT is the **compiler** — it takes that same model and compiles it into a native GPU executable that is 3–8× faster.

#### What optimizations does TensorRT apply?

**1. Layer Fusion**

Instead of running each operation as a separate GPU kernel (with memory reads/writes between each), TensorRT fuses compatible operations into single kernels:

```
Without TensorRT:
  Conv2D → [write to GPU memory] → BatchNorm → [write] → ReLU → [write]

With TensorRT (fused):
  Conv2D + BatchNorm + ReLU  →  [single kernel, one write]
```

This dramatically reduces GPU memory bandwidth usage and kernel launch overhead.

**2. Kernel Auto-Tuning**

For every layer in the network, TensorRT benchmarks dozens (sometimes hundreds) of different GPU kernel implementations and picks the absolute fastest one for *your specific GPU*. This is why:
- Compilation takes 5–20 minutes (it's running benchmarks)
- An engine built on RTX 3080 is faster than the same engine on RTX 3060 — they are literally different binary files

**3. Precision Optimization (FP16 / INT8)**

TensorRT can run layers in FP32, FP16, or INT8 precision. It selects the optimal precision per-layer to maximize speed while preserving accuracy within a user-defined tolerance.

**4. Memory Optimization**

TensorRT analyzes the entire graph and intelligently reuses GPU memory buffers across layers that are never active at the same time, minimizing peak VRAM consumption.

**5. Tensor Reformatting**

Data layout in GPU memory (NCHW vs NHWC) is automatically reformatted between layers to match what each kernel implementation expects — without any extra copy operations.

#### How much faster is TensorRT?

| Inference Mode | Typical Speedup vs. raw PyTorch |
|---|---|
| TensorRT FP32 | 1.5× – 2× faster |
| TensorRT FP16 | 3× – 5× faster |
| TensorRT INT8 | 5× – 8× faster |

For a ReID system processing 30+ person crops per frame, this speedup makes the difference between a pipeline that drops frames and one that runs comfortably in real time.

#### What is `trtexec`?

`trtexec` is the official command-line tool bundled with TensorRT for building and benchmarking engines. In this pipeline it handles the full compilation:

```bash
trtexec \
    --onnx=onnx_model/resnet50_reid.onnx \       # Input ONNX model
    --saveEngine=onnx_model/resnet50_reid.engine \# Output compiled engine
    --fp16 \                                      # Enable FP16 Tensor Core optimization
    --workspace=1024 \                            # 1GB scratch memory for auto-tuning
    --inputIOFormats=fp32:chw \                   # Input: FP32, Channel-Height-Width layout
    --outputIOFormats=fp32:chw                    # Output: FP32, Channel-Height-Width layout
```

#### Important: Engines are GPU-specific

⚠️ A TensorRT engine **cannot be shared between different GPU models**. An engine built on your RTX 3060 will produce incorrect or crashed results on someone else's RTX 4090.

```
Your RTX 3060 → trtexec → engine_A  (optimized for RTX 3060 silicon)
Friend's RTX 4090 → trtexec → engine_B  (optimized for RTX 4090 silicon)
```

The portable file is the `.onnx` (Step 1 output). Copy that to any machine and rebuild the engine there.

---

### 7. FP16 — Half Precision Inference

#### FP32 vs FP16: What is the difference?

Floating-point numbers are stored in binary. The number of bits determines the precision and range:

| Format | Bits | Relative Memory | Use Case |
|---|---|---|---|
| FP64 (double) | 64 | 4× | Scientific computing, high-precision math |
| FP32 (float) | 32 | 2× | Standard deep learning training |
| FP16 (half) | 16 | 1× | Fast GPU inference, training with mixed precision |
| INT8 | 8 | 0.5× | Ultra-fast inference (requires calibration) |

#### Why does FP16 work for neural network inference?

Neural networks are inherently **noise-tolerant**. The model was trained in FP32, but its predictions don't change meaningfully when weights are stored in FP16, because:

1. The model has learned **robust abstract features**, not exact numbers
2. Small numerical differences (e.g., `0.4312...` vs `0.43`) don't affect the final feature vector meaningfully
3. The error introduced by FP16 is well within the natural variance of real-world inputs (lighting changes, compression artifacts, etc.)

#### Why is FP16 faster?

Modern NVIDIA GPUs (Pascal architecture and newer) contain dedicated **Tensor Cores** — specialized hardware units designed for FP16 matrix multiplication. Compared to FP32 on standard CUDA cores:

- **2× less memory** needed to store the same model
- **2× more data fits in GPU cache** → fewer slow memory accesses
- **2× – 4× faster matrix multiplications** via Tensor Cores

For a ReID model that outputs a 256-dimensional feature vector, the accuracy difference between FP32 and FP16 inference is virtually undetectable in practice.

---

### 8. Docker and NVIDIA Container Toolkit

#### What is Docker?

**Docker** is a containerization platform. A **container** is a self-contained, isolated environment that packages an application together with all its exact dependencies — libraries, binaries, configuration — into a single portable unit.

```
Your Ubuntu 22.04 Machine
┌──────────────────────────────────────────────────┐
│  Docker Container (isolated environment)         │
│  ┌────────────────────────────────────────────┐  │
│  │  Ubuntu 20.04 base                         │  │
│  │  + Python 3.8                              │  │
│  │  + TensorFlow 2.11                         │  │
│  │  + CUDA 11.8 libraries                     │  │
│  │  + TAO Toolkit 5.2.0                       │  │
│  │  + All NVIDIA dependencies (exact versions)│  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

Without Docker, setting up the TAO Toolkit and all its exact dependency versions on your system would require installing specific CUDA versions, specific TensorFlow builds, and many NVIDIA libraries — with high risk of version conflicts. Docker reduces this to a single `docker pull` command.

#### What is the NVIDIA Container Toolkit?

By default, Docker containers are completely isolated from the host machine's hardware — including the GPU. The **NVIDIA Container Toolkit** (formerly called `nvidia-docker`) bridges this gap. It allows Docker containers to directly access the host's NVIDIA GPU and CUDA drivers.

```
TAO command inside container
         │
         │  (NVIDIA Container Toolkit creates the bridge)
         ▼
Host GPU Driver (installed on your system)
         │
         ▼
Physical GPU (RTX 3060, etc.)
```

This is why the Docker run command in `setup.sh` includes `--gpus all` — it tells Docker to expose all available NVIDIA GPUs to the container.

#### Why is Docker only needed for Step 1?

| Step | Tool | Distributed As | Docker Needed? |
|---|---|---|---|
| Step 1 (`.etlt` → `.onnx`) | NVIDIA TAO Toolkit | Docker image only | ✅ Yes |
| Step 2 (`.onnx` → `.engine`) | TensorRT / `trtexec` | pip package / deb | ❌ No |

TAO is only distributed as a Docker image by NVIDIA. TensorRT installs directly on your system via pip or the NVIDIA apt repository.

---

### 9. BNNeck — The Feature Neck

#### What is a "neck" in a neural network?

Modern ReID architectures split the network into three parts:
- **Backbone**: Extracts rich visual features from the image (ResNet-50)
- **Neck**: Processes and refines those backbone features
- **Head**: Produces the final output (feature vector or class prediction)

#### What is BNNeck?

**BNNeck** (Batch Normalization Neck) is a specific neck design from the paper *"Bag of Tricks and a Strong Baseline for Deep Person Re-identification"* (Luo et al., 2019). It applies a Batch Normalization layer after global average pooling to produce the final feature.

```
ResNet-50 Global Average Pooling Output
         │
         ▼
   Feature vector ft   (2048-dim, un-normalized)
         │
         ▼
   Batch Normalization
         │
         ▼
   Feature vector fi   (2048-dim → projected to 256-dim, normalized)
         │
         ▼
   Output: 256-dim ReID feature  ← What we use for person matching
```

#### Why does BNNeck improve ReID performance?

During training, two different loss functions are computed from different points:
- **Triplet Loss** is computed from `ft` (before BNNeck) — this loss pulls same-person features close and pushes different-person features apart in Euclidean space
- **ID Classification Loss** (Cross-Entropy) is computed from `fi` (after BNNeck) — this trains the model to correctly classify person IDs, learning discriminative features

Using both losses simultaneously forces the model to learn features that are both **discriminative for classification** and **well-clustered for retrieval**. The BNNeck is the architectural trick that allows both losses to be applied without conflicting with each other.

During **inference**, the BNNeck-normalized feature `fi` is used for cosine similarity comparisons — normalization makes all vectors unit-length, making cosine distance computation numerically stable and accurate.

---

## 🔄 Pipeline Overview

```
pretrained_model/
  resnet50_market1501_aicity156.etlt
              │
              │  bash setup.sh
              │  ├─ Docker pulls TAO image (nvcr.io/nvidia/tao/..., ~5GB, once)
              │  ├─ TAO decrypts .etlt using key "nvidia_tao"
              │  └─ TAO exports ResNet-50 architecture + weights to ONNX
              ▼
onnx_model/
  resnet50_reid.onnx                    ← Portable, inspectable, ~100 MB
              │
              │  bash convert_to_tensorrt.sh
              │  ├─ trtexec reads ONNX computational graph
              │  ├─ Fuses Conv + BN + ReLU layers into single kernels
              │  ├─ Auto-tunes GPU kernels for your specific GPU model
              │  └─ Compiles FP16-optimized binary engine
              ▼
onnx_model/
  resnet50_reid.engine                  ← GPU-specific, ~50-80 MB, maximum speed
              │
              ▼
  Your ReID / Tracking Pipeline  ✅
```

> ⚠️ **Docker is only needed for Step 1** — one time only. TensorRT in Step 2 runs natively.

---

## 📁 Project Structure

```
reid-etlt-converter/
│
├── pretrained_model/
│   ├── resnet50_market1501_aicity156.etlt   ← NVIDIA pre-trained ReID model (AES encrypted)
│   └── PUT_ETLT_FILE_HERE.txt               ← Placeholder reminder file
│
├── onnx_model/
│   └── OUTPUT_FILES_SAVED_HERE.txt          ← .onnx and .engine appear here after conversion
│
├── setup.sh                 ← Step 1: Runs TAO inside Docker → produces .onnx
├── convert_to_tensorrt.sh   ← Step 2: Runs trtexec natively → produces .engine
├── export.yaml              ← TAO export config (model architecture, input shape)
├── LICENSE                  ← Apache 2.0
└── README.md
```

---

## 🧰 Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Operating System | Ubuntu 18.04 | Ubuntu 20.04 / 22.04 |
| GPU | Any CUDA-capable NVIDIA GPU | RTX 3060 or better |
| VRAM | 4 GB | 8 GB+ |
| CUDA | 11.x | 12.x |
| TensorRT | 8.x | 8.6+ |
| Docker | 20.x | Latest |
| RAM | 8 GB | 16 GB |
| Free Disk Space | 10 GB | 20 GB (Docker image is ~5GB) |

### Verify your GPU is recognized

```bash
nvidia-smi
```

You should see your GPU name, driver version, and CUDA version listed.

### Install Docker (if not installed)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER    # Allow running docker without sudo
newgrp docker
```

### Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

If you see your GPU info printed from inside the container — you are ready.

### Install TensorRT

```bash
pip install tensorrt pycuda
```

Verify `trtexec` is accessible:

```bash
trtexec --help
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Abhishek242001/reid-etlt-converter.git
cd reid-etlt-converter
```

### 2. Verify the model file is present

```bash
ls pretrained_model/
# resnet50_market1501_aicity156.etlt
# PUT_ETLT_FILE_HERE.txt
```

> **Encryption Key**: `nvidia_tao` — already configured in `setup.sh`, no changes needed.

---

## 🔄 Conversion Steps

### Step 1: `.etlt` → `.onnx`  (One-time, requires Docker)

```bash
bash setup.sh
```

**What happens inside `setup.sh`, in order:**

1. Scans `pretrained_model/` for a `.etlt` file
2. Checks Docker is installed and running
3. Checks NVIDIA Container Toolkit — verifies GPU is accessible inside Docker
4. Pulls the TAO Toolkit image from NVIDIA's registry (~5GB, one-time download):
   ```
   nvcr.io/nvidia/tao/tao-toolkit:5.2.0-tf2.11.0
   ```
5. Runs the TAO export command inside the container:
   ```bash
   tao model re_identification export \
       -m /workspace/pretrained_model/resnet50_market1501_aicity156.etlt \
       -k nvidia_tao \
       --export_format ONNX \
       -e /workspace/export.yaml \
       -o /workspace/onnx_model/resnet50_reid.onnx
   ```
6. Verifies `onnx_model/resnet50_reid.onnx` was created and prints its file size

> ⏳ First run downloads ~5GB. Expect 5–15 minutes depending on internet speed. Subsequent runs are fast — Docker caches the image.

**Expected output:**
```
onnx_model/resnet50_reid.onnx   (~100 MB)
```

---

### Step 2: `.onnx` → TensorRT Engine  (No Docker required)

```bash
bash convert_to_tensorrt.sh
```

**What happens inside `convert_to_tensorrt.sh`, in order:**

1. Checks that `onnx_model/resnet50_reid.onnx` exists (Step 1 must complete first)
2. Checks `trtexec` is available — installs TensorRT via pip if not found
3. Runs `trtexec` to compile the TensorRT engine:
   ```bash
   trtexec \
       --onnx=onnx_model/resnet50_reid.onnx \
       --saveEngine=onnx_model/resnet50_reid.engine \
       --fp16 \
       --workspace=1024 \
       --inputIOFormats=fp32:chw \
       --outputIOFormats=fp32:chw
   ```
4. Verifies the engine file was created and prints its file size

**Flag-by-flag explanation:**

| Flag | Meaning |
|---|---|
| `--onnx` | Path to the input ONNX model |
| `--saveEngine` | Path to save the compiled TensorRT engine |
| `--fp16` | Enable FP16 half-precision using GPU Tensor Cores |
| `--workspace=1024` | Allow TensorRT up to 1024 MB of GPU memory during the build phase for kernel benchmarking |
| `--inputIOFormats=fp32:chw` | Input tensor is FP32 data type in Channel-Height-Width memory layout |
| `--outputIOFormats=fp32:chw` | Output tensor is FP32 in CHW layout |

> ⏳ Engine compilation takes **5–20 minutes**. TensorRT is internally benchmarking many kernel implementations. This is completely normal and only happens once.

**Expected output:**
```
onnx_model/resnet50_reid.engine   (~50–80 MB)
```

---

## ⚙️ Model Configuration

The `export.yaml` file specifies the exact model architecture passed to TAO during Step 1:

```yaml
inference_config:
  input_width: 128
  input_height: 256
  input_channels: 3

reid_config:
  backbone: resnet_50
  last_stride: 1
  pretrain_choice: imagenet
  input_channels: 3
  input_width: 128
  input_height: 256
  neck: bnneck
  feat_dim: 256
  neck_feat: after
  metric_loss_type: triplet
  with_center_loss: false
  with_flip_feature: false
  label_smooth: true
```

**Parameter explanations:**

| Parameter | Value | Explanation |
|---|---|---|
| `input_width` | 128 | Input image width — person crops are narrow (portrait orientation) |
| `input_height` | 256 | Input image height — twice the width, standard ReID crop shape |
| `input_channels` | 3 | RGB color image |
| `backbone` | resnet_50 | Feature extractor: 50-layer Residual Network |
| `last_stride` | 1 | The last ResNet stage uses stride=1 instead of 2, preserving more spatial detail in features |
| `pretrain_choice` | imagenet | The ResNet-50 backbone was first pre-trained on ImageNet, then fine-tuned for ReID |
| `neck` | bnneck | Batch Normalization Neck — normalizes the feature vector before output |
| `feat_dim` | 256 | Final output feature vector dimension (256 numbers per person) |
| `neck_feat` | after | Use the feature *after* BNNeck (normalized) for inference |
| `metric_loss_type` | triplet | Trained with Triplet Loss — brings same-person embeddings close, pushes different-person embeddings apart |
| `with_center_loss` | false | Center Loss not used (simpler training regime) |
| `label_smooth` | true | Label smoothing applied during ID classification — prevents the model from becoming overconfident on training identities |

---

## ❓ Frequently Asked Questions

**Q: What is a `.etlt` file and why is it encrypted?**

A: `.etlt` is NVIDIA's proprietary encrypted model format from the TAO Toolkit. Encryption protects NVIDIA's IP and ensures models are used through the official export pipeline. For public community models like this one, NVIDIA openly publishes the decryption key (`nvidia_tao`).

---

**Q: Do I need a data center GPU?**

A: No. This pipeline is specifically designed for **desktop GPUs**. Any CUDA-capable NVIDIA GPU works. An RTX 3060 (6GB VRAM) is a comfortable minimum. Even older cards like GTX 1080 Ti work for both conversion and inference.

---

**Q: Why is Docker needed only for Step 1?**

A: TAO Toolkit is a large software suite that NVIDIA ships exclusively as a Docker container. It bundles TensorFlow, CUDA libraries, and NVIDIA's proprietary model tools in a pre-configured environment. After the one-time ONNX conversion, Docker is done — TensorRT in Step 2 installs directly on your system via pip.

---

**Q: Engine compilation takes 15+ minutes. Is something wrong?**

A: No — this is completely normal. TensorRT is benchmarking dozens of GPU kernel implementations for every layer in the network to find the fastest option for your specific GPU. This only happens once. The resulting `.engine` file is saved and reused forever.

---

**Q: Can I copy the `.engine` file to another computer?**

A: The `.engine` file is **GPU-specific** and cannot be used on a different GPU model. However, the `.onnx` file (Step 1 output) is fully portable. Copy the `.onnx` to any machine and run Step 2 there to build a compatible engine for that GPU.

---

**Q: What input format does the engine expect?**

A: The engine expects:
- **Shape**: `(batch_size, 3, 256, 128)` — Batch × Channels × Height × Width
- **Data type**: FP32 (even though the engine runs in FP16 internally)
- **Layout**: CHW (Channel-first — standard for PyTorch and TensorRT)
- **Normalization**: ImageNet mean/std normalization is expected (subtract mean, divide by std)

---

**Q: Can I run inference using just the `.onnx` file without TensorRT?**

A: Yes. Install `onnxruntime-gpu` and load the ONNX file directly. It will be slower than TensorRT but still GPU-accelerated — useful for testing or on systems where TensorRT isn't available:

```bash
pip install onnxruntime-gpu
```

---

**Q: What datasets was this model trained on?**

A: The model was trained on two datasets:
- **Market-1501**: 32,668 images of 1,501 identities from 6 cameras (Tsinghua University)
- **AICityChallenge Track 1**: Data from NVIDIA's AI City Challenge competition on vehicle/person ReID

---

## 🔧 Troubleshooting

**`trtexec: command not found`**

```bash
pip install tensorrt pycuda
# Or locate trtexec manually:
find / -name "trtexec" 2>/dev/null
```

---

**`no matching manifest for linux/arm64`**

The TAO Toolkit Docker image is x86_64 (Intel/AMD) only. This pipeline does not support ARM processors (Apple Silicon, Raspberry Pi).

---

**GPU not accessible inside Docker**

```bash
sudo apt-get remove --purge nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

---

**Out of GPU memory during engine build**

Reduce the workspace size in `convert_to_tensorrt.sh`:

```bash
--workspace=512    # Reduce from 1024 to 512 MB
```

---

**`onnx_model/resnet50_reid.onnx` not found when running Step 2**

Step 1 must complete successfully first:

```bash
bash setup.sh                  # Must run this first
bash convert_to_tensorrt.sh    # Then this
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **Apache 2.0 License** — see the [LICENSE](./LICENSE) file for full terms.

Apache 2.0 allows you to freely use, modify, and distribute this code — including in commercial projects — as long as you include the original license and copyright notice.

---

## 🙏 Acknowledgements

- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) — Pre-trained ReID models and export tooling
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — GPU inference optimization engine
- [ONNX](https://onnx.ai/) — Open Neural Network Exchange format
- [Market-1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) — Zheng et al., Tsinghua University
- [ResNet Paper](https://arxiv.org/abs/1512.03385) — *"Deep Residual Learning for Image Recognition"*, He et al., CVPR 2016
- [BNNeck Paper](https://arxiv.org/abs/1906.08332) — *"Bag of Tricks and a Strong Baseline for Deep Person Re-identification"*, Luo et al., 2019

---
