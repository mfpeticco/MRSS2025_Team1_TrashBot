#!/usr/bin/env bash
#
# inference_ncnn.sh
# Press [space] to advance, [q] to quit, [r] to repeat current image.

DEMO="ncnn_test/build/demo"
PARAM="ncnn_test/best-opt.param"
BIN="ncnn_test/best-opt.bin"
IMG_DIR="data/utd2-7/valid/images"

# Sanity checks
if [ ! -x "$DEMO" ]; then
  echo "ERROR: demo not found or not executable at $DEMO"
  echo "Make sure to build the project first: cd ncnn_test && mkdir -p build && cd build && cmake .. && make"
  exit 1
fi
if [ ! -f "$PARAM" ] || [ ! -f "$BIN" ]; then
  echo "ERROR: model files missing:"
  echo "  $PARAM"
  echo "  $BIN"
  exit 1
fi

shopt -s nullglob
images=( "$IMG_DIR"/*.jpg "$IMG_DIR"/*.jpeg "$IMG_DIR"/*.png )
if [ ${#images[@]} -eq 0 ]; then
  echo "No images in $IMG_DIR"
  exit 0
fi

index=0
total=${#images[@]}

echo "YOLOv11n Trash Detection - NCNN Inference"
echo "=========================================="
echo "Classes: Bio (Green), Rov (Blue), Trash (Red)"
echo "Controls: [space] = next, [r] = repeat, [q] = quit"
echo

while (( index < total )); do
  img="${images[index]}"
  clear
  echo "[$((index+1))/$total] Processing: $(basename "$img")"
  echo "----------------------------------------"
  
  # Run inference
  "$DEMO" "$PARAM" "$BIN" "$img"
  
  echo
  echo "Controls: [space] = next, [r] = repeat, [q] = quit"
  echo -n "Your choice: "
  
  # Read one character
  IFS= read -r -n1 key
  echo
  
  case "$key" in
    [qQ])
      echo "Exiting."
      exit 0
      ;;
    [rR])
      # Repeat current image
      continue
      ;;
    ' ')
      # Advance to next image
      ((index++))
      ;;
    *)
      # Any other key, treat as next
      ((index++))
      ;;
  esac
done

echo "All images processed."