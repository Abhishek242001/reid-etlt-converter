#!/bin/bash

# =============================================================================
# ReID Model Conversion: .etlt → .onnx
# Using NVIDIA TAO Toolkit (Docker) — one time only
# =============================================================================

set -e  # Exit on any error

# ── Colors for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Config ───────────────────────────────────────────────────────────────────
TAO_IMAGE="nvcr.io/nvidia/tao/tao-toolkit:5.2.0-tf2.11.0"
MODEL_KEY="nvidia_tao"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_DIR="$SCRIPT_DIR/pretrained_model"
ONNX_DIR="$SCRIPT_DIR/onnx_model"
EXPORT_YAML="$SCRIPT_DIR/export.yaml"

# =============================================================================
# STEP 0: Print banner
# =============================================================================
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   NVIDIA ReID Model Conversion: .etlt → .onnx              ${NC}"
echo -e "${BLUE}   TAO Toolkit — One Time Conversion                        ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# =============================================================================
# STEP 1: Check pretrained_model folder has .etlt file
# =============================================================================
echo -e "${YELLOW}[Step 1/6] Checking for .etlt model file...${NC}"

ETLT_FILE=$(find "$PRETRAINED_DIR" -name "*.etlt" | head -n 1)

if [ -z "$ETLT_FILE" ]; then
    echo -e "${RED}ERROR: No .etlt file found in pretrained_model/ folder!${NC}"
    echo ""
    echo "  Please copy your .etlt file into:"
    echo "  → $PRETRAINED_DIR/"
    echo ""
    echo "  Example:"
    echo "  cp resnet50_market1501_aicity156.etlt $PRETRAINED_DIR/"
    echo ""
    exit 1
fi

ETLT_FILENAME=$(basename "$ETLT_FILE")
echo -e "${GREEN}  ✓ Found: $ETLT_FILENAME${NC}"

# =============================================================================
# STEP 2: Check Docker is installed
# =============================================================================
echo ""
echo -e "${YELLOW}[Step 2/6] Checking Docker installation...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed!${NC}"
    echo ""
    echo "  Install Docker with:"
    echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "  sudo sh get-docker.sh"
    echo ""
    exit 1
fi
echo -e "${GREEN}  ✓ Docker found: $(docker --version)${NC}"

# =============================================================================
# STEP 3: Check NVIDIA Docker runtime
# =============================================================================
echo ""
echo -e "${YELLOW}[Step 3/6] Checking NVIDIA Container Toolkit...${NC}"

if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA Container Toolkit not working!${NC}"
    echo ""
    echo "  Install with:"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    echo ""
    exit 1
fi
echo -e "${GREEN}  ✓ NVIDIA Docker runtime is working${NC}"

# =============================================================================
# STEP 4: Pull TAO Toolkit Docker image
# =============================================================================
echo ""
echo -e "${YELLOW}[Step 4/6] Pulling TAO Toolkit Docker image...${NC}"
echo "  Image: $TAO_IMAGE"
echo "  (This may take a few minutes on first run ~5GB)"
echo ""

docker pull "$TAO_IMAGE"
echo -e "${GREEN}  ✓ TAO Toolkit image ready${NC}"

# =============================================================================
# STEP 5: Run TAO export inside Docker (.etlt → .onnx)
# =============================================================================
echo ""
echo -e "${YELLOW}[Step 5/6] Running TAO export: .etlt → .onnx...${NC}"
echo "  Input  : pretrained_model/$ETLT_FILENAME"
echo "  Output : onnx_model/resnet50_reid.onnx"
echo "  Key    : $MODEL_KEY"
echo ""

docker run --gpus all --rm \
    -v "$SCRIPT_DIR":/workspace \
    "$TAO_IMAGE" \
    tao model re_identification export \
        -m /workspace/pretrained_model/"$ETLT_FILENAME" \
        -k "$MODEL_KEY" \
        --export_format ONNX \
        -e /workspace/export.yaml \
        -o /workspace/onnx_model/resnet50_reid.onnx

# =============================================================================
# STEP 6: Verify output
# =============================================================================
echo ""
echo -e "${YELLOW}[Step 6/6] Verifying output...${NC}"

ONNX_FILE="$ONNX_DIR/resnet50_reid.onnx"

if [ -f "$ONNX_FILE" ]; then
    SIZE=$(du -sh "$ONNX_FILE" | cut -f1)
    echo -e "${GREEN}  ✓ ONNX model created successfully!${NC}"
    echo -e "${GREEN}  ✓ Location : onnx_model/resnet50_reid.onnx${NC}"
    echo -e "${GREEN}  ✓ File size: $SIZE${NC}"
else
    echo -e "${RED}ERROR: ONNX file was not created. Check TAO output above.${NC}"
    exit 1
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}  CONVERSION COMPLETE!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "  Your ONNX model is ready at:"
echo "  → $ONNX_FILE"
echo ""
echo "  Next steps:"
echo "  1. Run convert_to_tensorrt.sh  to build TensorRT .engine"
echo "  2. Plug engine into your OSNetFeatureExtractor"
echo ""
echo -e "${YELLOW}  Docker is no longer needed after this point.${NC}"
echo ""
