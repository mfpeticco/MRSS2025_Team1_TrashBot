#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

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

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " model.param model.bin image.jpg\n";
        return 1;
    }
    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* img_path = argv[3];

    // Load image
    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) {
        std::cerr << "failed to read image: " << img_path << "\n";
        return 1;
    }

    // Store original dimensions
    int orig_width = bgr.cols;
    int orig_height = bgr.rows;

    // Prepare input
    const int target_size = 416;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);

    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(0, norm_vals);

    // Load network
    ncnn::Net yolov11n;
    yolov11n.opt.use_vulkan_compute = false;
    yolov11n.load_param(param_path);
    yolov11n.load_model(bin_path);

    // Run inference
    ncnn::Extractor ex = yolov11n.create_extractor();
    ex.input("images", in);
    ncnn::Mat out;
    ex.extract("output0", out);

    // Debug output shape
    std::cout << "Output shape: " << out.w << " x " << out.h << " x " << out.c << std::endl;

    // Process detections - YOLOv11n output format: [3549, 38]
    // where 38 = 4*16 (DFL bbox regression) + 3 (classes) + 19 (other outputs)
    std::vector<Detection> proposals;
    const float confidence_threshold = 0.25f;  // Lower threshold to start
    const float nms_threshold = 0.4f;
    const int num_classes = 3;
    
    int num_detections = out.h; // Should be 3549
    
    // Calculate scale factors
    float scale_x = (float)orig_width / target_size;
    float scale_y = (float)orig_height / target_size;
    
    // Generate anchor points for 416x416 input
    std::vector<std::pair<float, float>> anchor_points;
    std::vector<int> strides = {8, 16, 32}; // YOLOv11n strides
    std::vector<int> grid_sizes = {52, 26, 13}; // 416/8, 416/16, 416/32
    
    for (int i = 0; i < 3; i++) {
        int stride = strides[i];
        int grid_size = grid_sizes[i];
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                float anchor_x = (x + 0.5f) * stride;
                float anchor_y = (y + 0.5f) * stride;
                anchor_points.push_back({anchor_x, anchor_y});
            }
        }
    }
    
    std::cout << "Generated " << anchor_points.size() << " anchor points" << std::endl;
    
    for (int i = 0; i < num_detections && i < (int)anchor_points.size(); i++) {
        const float* ptr = out.row(i);
        
        // Extract DFL bbox predictions (first 64 values = 4*16)
        float bbox_pred[4] = {0};
        for (int j = 0; j < 4; j++) {
            // Apply softmax to the 16 values for each bbox coordinate
            float sum = 0.0f;
            float max_val = ptr[j * 16];
            for (int k = 1; k < 16; k++) {
                if (ptr[j * 16 + k] > max_val) {
                    max_val = ptr[j * 16 + k];
                }
            }
            
            for (int k = 0; k < 16; k++) {
                float exp_val = expf(ptr[j * 16 + k] - max_val);
                sum += exp_val;
                bbox_pred[j] += exp_val * k;
            }
            bbox_pred[j] /= sum;
        }
        
        // Convert DFL predictions to actual coordinates
        float anchor_x = anchor_points[i].first;
        float anchor_y = anchor_points[i].second;
        
        float x1 = anchor_x - bbox_pred[0];
        float y1 = anchor_y - bbox_pred[1];
        float x2 = anchor_x + bbox_pred[2];
        float y2 = anchor_y + bbox_pred[3];
        
        // Scale to original image size
        x1 *= scale_x;
        y1 *= scale_y;
        x2 *= scale_x;
        y2 *= scale_y;
        
        // Convert to x,y,w,h format
        float x = x1;
        float y = y1;
        float w = x2 - x1;
        float h = y2 - y1;
        
        // Skip invalid boxes
        if (w <= 0 || h <= 0) continue;
        
        // Extract class predictions (starting at index 64)
        float max_confidence = 0.0f;
        int best_class = 0;
        
        for (int j = 0; j < num_classes; j++) {
            float confidence = 1.0f / (1.0f + expf(-ptr[64 + j])); // Sigmoid
            if (confidence > max_confidence) {
                max_confidence = confidence;
                best_class = j;
            }
        }
        
        // Filter by confidence
        if (max_confidence > confidence_threshold) {
            // Clamp to image boundaries
            x = std::max(0.0f, std::min(x, (float)orig_width - 1));
            y = std::max(0.0f, std::min(y, (float)orig_height - 1));
            w = std::max(1.0f, std::min(w, (float)orig_width - x));
            h = std::max(1.0f, std::min(h, (float)orig_height - y));
            
            proposals.push_back({cv::Rect_<float>(x, y, w, h), best_class, max_confidence});
        }
    }
    
    std::cout << "Found " << proposals.size() << " proposals before NMS" << std::endl;
    
    // Sort by confidence
    std::sort(proposals.begin(), proposals.end(), [](const Detection& a, const Detection& b) {
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
        
        std::string label = cv::format("%s: %.2f", class_names[det.class_id].c_str(), det.confidence);
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
    
    cv::imwrite("result.jpg", bgr);
    std::cout << "Detection results saved to result.jpg\n";
    cv::imshow("Detections", bgr);
    cv::waitKey(0);

    return 0;
}