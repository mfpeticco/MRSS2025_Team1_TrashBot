#!/usr/bin/env bash
# inference-ncnn.sh
# pick one image from val set, run detect, write output.jpg

MODEL_DIR="ncnn_test"
BINARY="ncnn_test/build/demo"   # path to your compiled main.cpp binary
IMG_DIR="data/utd2-7/valid/images"

# find the first image (jpg/png) in the directory
IMG_PATH=$(find "${IMG_DIR}" -type f \( -iname "*.jpg" -o -iname "*.png" \) | head -n1)

if [ -z "${IMG_PATH}" ]; then
    echo "❌ No image files found in ${IMG_DIR}" >&2
    exit 1
fi

echo "🔍 Running inference on: ${IMG_PATH}"
"${BINARY}" "${IMG_PATH}"

if [ $? -eq 0 ]; then
    echo "✅ Done! See output.jpg in $(pwd)"
else
    echo "❌ Inference failed" >&2
    exit 1
fi