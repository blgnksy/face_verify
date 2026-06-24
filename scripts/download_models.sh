#!/bin/bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "=== Downloading face detection model (YuNet) ==="
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
if [ ! -f "$MODELS_DIR/face_detection_yunet_2023mar.onnx" ]; then
    wget -q --show-progress -O "$MODELS_DIR/face_detection_yunet_2023mar.onnx" "$YUNET_URL"
    echo "  ✓ YuNet downloaded"
else
    echo "  ✓ YuNet already exists"
fi

echo ""
echo "=== Downloading face recognition model (ArcFace MobileFaceNet) ==="
echo ""
echo "The ArcFace model (w600k_mbf.onnx) must be downloaded from InsightFace."
echo "It's distributed in a zip archive."
echo ""
echo "Option 1 — From InsightFace model zoo (recommended):"
echo "  wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
echo "  unzip buffalo_sc.zip -d /tmp/buffalo_sc"
echo "  cp /tmp/buffalo_sc/w600k_mbf.onnx $MODELS_DIR/"
echo ""
echo "Option 2 — If you have Python insightface installed:"
echo "  python3 -c \""
echo "  import insightface"
echo "  app = insightface.app.FaceAnalysis(name='buffalo_sc')"
echo "  # Model will be cached in ~/.insightface/models/buffalo_sc/"
echo "  \""
echo "  cp ~/.insightface/models/buffalo_sc/w600k_mbf.onnx $MODELS_DIR/"
echo ""

# Try automatic download
ARCFACE_ZIP_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
if [ ! -f "$MODELS_DIR/w600k_mbf.onnx" ]; then
    echo "Attempting automatic download..."
    TMPDIR=$(mktemp -d)
    if wget -q --show-progress -O "$TMPDIR/buffalo_sc.zip" "$ARCFACE_ZIP_URL" 2>/dev/null; then
        unzip -o -j "$TMPDIR/buffalo_sc.zip" "w600k_mbf.onnx" -d "$MODELS_DIR/" 2>/dev/null || true
        rm -rf "$TMPDIR"
        if [ -f "$MODELS_DIR/w600k_mbf.onnx" ]; then
            echo "  ✓ ArcFace downloaded"
        else
            echo "  ✗ Extraction failed — download manually (see above)"
        fi
    else
        rm -rf "$TMPDIR"
        echo "  ✗ Download failed — download manually (see above)"
    fi
else
    echo "  ✓ ArcFace already exists"
fi

echo ""
echo "=== Model summary ==="
ls -lh "$MODELS_DIR"/*.onnx 2>/dev/null || echo "No models found in $MODELS_DIR/"
