#!/bin/bash

# =============================================================================
# TensorRT Engine Build: .onnx → .engine
# No Docker needed — pure desktop Python/TensorRT
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_FILE="$SCRIPT_DIR/onnx_model/resnet50_reid.onnx"
ENGINE_FILE="$SCRIPT_DIR/onnx_model/resnet50_reid.engine"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   TensorRT Engine Build: .onnx → .engine                  ${NC}"
echo -e "${BLUE}   No Docker needed — runs on your desktop GPU             ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ── Check ONNX file exists ───────────────────────────────────────────────────
echo -e "${YELLOW}[Step 1/3] Checking ONNX model...${NC}"
if [ ! -f "$ONNX_FILE" ]; then
    echo -e "${RED}ERROR: onnx_model/resnet50_reid.onnx not found!${NC}"
    echo "  Please run setup.sh first to convert .etlt → .onnx"
    exit 1
fi
echo -e "${GREEN}  ✓ Found: onnx_model/resnet50_reid.onnx${NC}"

# ── Check trtexec ────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[Step 2/3] Checking TensorRT...${NC}"
if ! command -v trtexec &> /dev/null; then
    echo -e "${YELLOW}  trtexec not found in PATH, trying pip install...${NC}"
    pip install tensorrt pycuda cuda-python --quiet
fi
echo -e "${GREEN}  ✓ TensorRT ready${NC}"

# ── Build engine ─────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[Step 3/3] Building TensorRT engine (FP16)...${NC}"
echo "  Input  : onnx_model/resnet50_reid.onnx"
echo "  Output : onnx_model/resnet50_reid.engine"
echo "  Mode   : FP16 (fastest on desktop GPU)"
echo ""

trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 \
    --workspace=1024 \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw

# ── Verify ───────────────────────────────────────────────────────────────────
if [ -f "$ENGINE_FILE" ]; then
    SIZE=$(du -sh "$ENGINE_FILE" | cut -f1)
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}  ENGINE BUILD COMPLETE!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo -e "${GREEN}  ✓ Location : onnx_model/resnet50_reid.engine${NC}"
    echo -e "${GREEN}  ✓ File size: $SIZE${NC}"
    echo ""
    echo "  Next step:"
    echo "  Copy resnet50_reid.engine to your project and"
    echo "  update OSNetFeatureExtractor to load this engine."
    echo ""
else
    echo -e "${RED}ERROR: Engine file was not created.${NC}"
    exit 1
fi
