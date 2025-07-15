// main.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <float.h>

struct Detection
{
    cv::Rect_<float> box;
    int class_id;
    float confidence;
};

static void nms_sorted_bboxes(const std::vector<Detection>& detections, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = detections.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = detections[i].box.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Detection& a = detections[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Detection& b = detections[picked[j]];
            float inter_area = (a.box & b.box).area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static float softmax_sum(const float* src, float* dst, int length)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static float clamp(float val, float min_val, float max_val)
{
    return val > min_val ? (val < max_val ? val : max_val) : min_val;
}

// Helper function to apply sigmoid activation
static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    const char* img_path = argv[1];
    const char* param_path = "ncnn_test/best-opt.param";
    const char* bin_path = "ncnn_test/best-opt.bin";

    // Load image
    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) {
        std::cerr << "failed to read image: " << img_path << "\n";
        return -1;
    }

    // Store original dimensions
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // Load network
    ncnn::Net yolov11n;
    yolov11n.opt.use_vulkan_compute = false;
    yolov11n.opt.num_threads = 4;
    if (yolov11n.load_param(param_path) != 0 || yolov11n.load_model(bin_path) != 0) {
        std::cerr << "Failed to load NCNN model files!\n";
        return -1;
    }

    // Letterbox preprocessing - YOLOv11n was trained with letterbox
    const int target_size = 416;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    // Calculate letterbox parameters
    float scale = std::min((float)target_size / img_w, (float)target_size / img_h);
    int scaled_w = std::round(img_w * scale);
    int scaled_h = std::round(img_h * scale);
    
    // Calculate padding
    int pad_w = target_size - scaled_w;
    int pad_h = target_size - scaled_h;
    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;
    
    std::cout << "Letterbox info: scale=" << scale 
              << ", padding [top=" << top << ", bottom=" << bottom 
              << ", left=" << left << ", right=" << right << "]" << std::endl;

    // Resize while maintaining aspect ratio
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(scaled_w, scaled_h));
    
    // Add letterbox padding
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Save the letterboxed image for debugging
    cv::imwrite("letterboxed_input.jpg", padded);

    // Convert to NCNN format
    ncnn::Mat in = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB, target_size, target_size);
    
    // Normalize
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(0, norm_vals);

    // Run inference
    ncnn::Extractor ex = yolov11n.create_extractor();
    ex.input("images", in);
    ncnn::Mat out;
    ex.extract("output0", out);

    // Debug output shape
    std::cout << "Output shape: " << out.w << " x " << out.h << " x " << out.c << std::endl;

    // Process detections
    std::vector<Detection> proposals;
    const int num_classes = 3;
    
    for (int i = 0; i < out.h; i++) {
        const float* ptr = out.row(i);
        
        // IMPORTANT: The first 4 values are normalized bbox coordinates (0-1)
        // We need to scale them to image dimensions
        float x1 = ptr[0] * target_size;  // Already normalized (0-1)
        float y1 = ptr[1] * target_size;
        float x2 = ptr[2] * target_size;
        float y2 = ptr[3] * target_size;
        
        // Skip invalid boxes
        if (x2 <= x1 || y2 <= y1) continue;
        
        // Find class with highest confidence (need to apply sigmoid)
        float max_confidence = 0.0f;
        int best_class = -1;
        
        for (int j = 0; j < num_classes; j++) {
            // Apply sigmoid to class score
            float confidence = sigmoid(ptr[4 + j]);
            
            if (confidence > max_confidence) {
                max_confidence = confidence;
                best_class = j;
            }
        }
        
        // Debug the first few detections
        if (i < 5) {
            std::cout << "Detection " << i << ": "
                      << "bbox=" << x1 << "," << y1 << "," << x2 << "," << y2
                      << " confidence=" << max_confidence 
                      << " class=" << best_class << std::endl;
        }
        
        // Filter by confidence
        if (max_confidence > prob_threshold) {
            // Remove letterbox padding
            x1 -= left;
            y1 -= top;
            x2 -= left;
            y2 -= top;
            
            // Scale back to original image dimensions
            x1 /= scale;
            y1 /= scale;
            x2 /= scale;
            y2 /= scale;
            
            // Clamp to original image bounds
            x1 = std::max(0.0f, std::min(x1, (float)img_w));
            y1 = std::max(0.0f, std::min(y1, (float)img_h));
            x2 = std::max(0.0f, std::min(x2, (float)img_w));
            y2 = std::max(0.0f, std::min(y2, (float)img_h));
            
            float width = x2 - x1;
            float height = y2 - y1;
            
            // Only add valid boxes
            if (width > 1 && height > 1) {
                Detection det;
                det.box = cv::Rect_<float>(x1, y1, width, height);
                det.class_id = best_class;
                det.confidence = max_confidence;
                proposals.push_back(det);
            }
        }
    }
    
    std::cout << "Found " << proposals.size() << " proposals before NMS" << std::endl;
    
    // Sort by confidence
    std::sort(proposals.begin(), proposals.end(), 
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    // Apply NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    // Draw results
    const std::vector<std::string> class_names = {"Bio", "Rov", "Trash"};
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),    // Bio - Green
        cv::Scalar(255, 0, 0),    // Rov - Blue  
        cv::Scalar(0, 0, 255)     // Trash - Red
    };
    
    for (size_t i = 0; i < picked.size(); ++i) {
        const Detection& det = proposals[picked[i]];
        
        cv::rectangle(bgr, det.box, colors[det.class_id], 2);
        
        std::string label = cv::format("%s: %.3f", class_names[det.class_id].c_str(), det.confidence);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int label_y = std::max((int)det.box.y, label_size.height);
        cv::rectangle(bgr, cv::Point(det.box.x, label_y - label_size.height), 
                      cv::Point(det.box.x + label_size.width, label_y + baseLine), 
                      colors[det.class_id], -1);
        
        cv::putText(bgr, label, cv::Point(det.box.x, label_y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        std::cout << "Detection: " << class_names[det.class_id]
                  << " | Confidence: " << det.confidence
                  << " | Box: [" << det.box.x << ", " << det.box.y << ", " << det.box.width << ", " << det.box.height << "]\n";
    }
    
    std::cout << "Final detections: " << picked.size() << std::endl;
    
    cv::imwrite("output.jpg", bgr);
    std::cout << "Detection results saved to output.jpg\n";
    
    return 0;
}