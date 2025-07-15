rm -rf build
mkdir -p build
cd build
cmake \
  -DOpenCV_DIR=/opt/homebrew/opt/opencv/lib/cmake/opencv4 \
  -DNCNN_ROOT=/opt/homebrew/opt/ncnn \
  ..
make -j4