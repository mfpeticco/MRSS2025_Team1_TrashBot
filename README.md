Steps taken:

1. Use train.py, which in it contains code to pull yolo11n.pt and fine tune the model on the data/utd2-7 dataset. Note oceantrash_dataset is the same thing - just the output of the standalone dataset downloader. Training gives models/yolov11n_trash_detection. 
2. You can test the model using inference.py. However this just runs on your computer.

Set up NCNN model and optimize it for the raspberry pi:
1. in main project dir, run ncnn_simple_export.py to get onnx and ncnn versions of the models. Make sure both are installed, i used brew since im on an apple silicon mac
2. cd models/yolov11n_trash_detection/weights
3. python -m onnxsim best.onnx best-simplified.onnx (need pip install onnx onnx-simplifier)
4. onnx2ncnn best-simplified.onnx best.param best.bin
5. ncnnoptimize best.param best.bin best-opt.param best-opt.bin 0