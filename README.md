# 🔍 ReID ETLT Converter

> **NVIDIA ReID Model Conversion Pipeline** — Convert `.etlt` encrypted models to `.onnx` and then to a **TensorRT engine** for high-performance Person Re-Identification (ReID) on desktop GPUs.

---

## 📌 What Is This?

This repository provides a step-by-step pipeline to convert NVIDIA's encrypted **ReID (Re-Identification)** model (`.etlt` format) into a **TensorRT engine** that can run efficiently on a desktop GPU.

**Person Re-Identification (ReID)** is a computer vision task where a model identifies the same person across different camera views or video frames — even if the person looks slightly different due to angle, lighting, or clothing.

NVIDIA provides pre-trained ReID models in a proprietary `.etlt` format. This pipeline helps you:
1. Decrypt and convert `.etlt` → `.onnx`
2. Optimize `.onnx` → **TensorRT engine** (`.engine` / `.trt`)

---

## 🧠 Pipeline Overview

```
NVIDIA ReID Model
      │
      ▼
  [ .etlt file ]          ← Encrypted model from NVIDIA
      │
      │  Step 1: tao-converter / trtexec
      ▼
  [ .onnx file ]          ← Standard, open format
      │
      │  Step 2: trtexec / Python script
      ▼
  [ TensorRT Engine ]     ← Optimized for your GPU
      │
      ▼
  Run Inference ✅
```

---

## 🧰 Requirements

Before you begin, make sure you have the following installed:

| Dependency | Version |
|---|---|
| Python | 3.8+ |
| CUDA | 11.x or 12.x |
| TensorRT | 8.x or later |
| NVIDIA TAO Toolkit / `tao-converter` | Latest |
| `onnx` Python package | `pip install onnx` |
| `onnxruntime-gpu` | `pip install onnxruntime-gpu` |

> ⚠️ Make sure your GPU drivers are up to date. Run `nvidia-smi` to verify your GPU is detected.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Abhishek242001/reid-etlt-converter.git
cd reid-etlt-converter
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔄 Conversion Steps

### Step 1: Convert `.etlt` → `.onnx`

Use the `tao-converter` tool from NVIDIA to decrypt and export the model:

```bash
tao-converter \
  -k <your_encryption_key> \
  -d 3,256,128 \
  -p input,1x3x256x128,4x3x256x128,8x3x256x128 \
  -e output/reid_model.engine \
  -t fp16 \
  -o output/reid_model.onnx \
  input/reid_model.etlt
```

> 📝 **Note:** Replace `<your_encryption_key>` with the key provided by NVIDIA for the model.  
> The input shape `3,256,128` means: 3 color channels, 256 height, 128 width — standard for ReID models.

---

### Step 2: Convert `.onnx` → TensorRT Engine

```bash
trtexec \
  --onnx=output/reid_model.onnx \
  --saveEngine=output/reid_model.engine \
  --fp16 \
  --minShapes=input:1x3x256x128 \
  --optShapes=input:4x3x256x128 \
  --maxShapes=input:8x3x256x128
```

> ⚡ `--fp16` enables half-precision mode, which speeds up inference on modern NVIDIA GPUs (RTX series, etc.).

---

### Step 3: Run Inference

Once you have the `.engine` file, you can use it for inference in your tracking or surveillance pipeline:

```python
import tensorrt as trt
# Load engine and run inference
# (See inference.py for a complete working example)
```

Refer to [`inference.py`](./inference.py) for a full working inference script.

---

## 📁 Project Structure

```
reid-etlt-converter/
│
├── input/                  # Place your .etlt model here
├── output/                 # Generated .onnx and .engine files go here
├── scripts/
│   ├── convert_etlt.sh     # Shell script for Step 1
│   └── build_engine.sh     # Shell script for Step 2
├── inference.py            # Example inference script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ❓ Frequently Asked Questions

**Q: What is a `.etlt` file?**  
A: It's an encrypted TLT (Transfer Learning Toolkit) model format used by NVIDIA. You need the correct encryption key to convert it.

**Q: Do I need a data center GPU?**  
A: No! This pipeline is designed to work on **desktop GPUs** (e.g., RTX 3060, RTX 4080). Any CUDA-capable NVIDIA GPU should work.

**Q: What is TensorRT?**  
A: TensorRT is NVIDIA's deep learning inference optimizer. It takes a trained model and compiles it into a highly optimized engine for your specific GPU, making inference much faster.

**Q: Why convert to ONNX first?**  
A: ONNX (Open Neural Network Exchange) is an open standard format. Converting through ONNX gives you flexibility — you can inspect the model, validate it, or use it with other frameworks before building the TensorRT engine.

---

## 🤝 Contributing

Contributions are welcome! If you run into issues or want to add support for new model versions:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push and open a Pull Request

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgements

- [NVIDIA TAO Toolkit](https://developer.nvidia.com/tao-toolkit) — for the pre-trained ReID models
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — for the inference optimization engine
- [ONNX](https://onnx.ai/) — for the open model exchange format

---
