#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
namespace fs = std::filesystem;


// Constants for YOLOv8/v11 Nano models
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;





int detection(cv::dnn::Net& net, const cv::Mat& frame, const string& frame_name) {

    // 1. Load the network
    // cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    
    // CPU Optimization
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 2. Load test image
    // cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "Error: Image not found.\n";
        return -1;
    }

    // 3. Preprocessing (Convert image to 4D Blob)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    // 4. Run Forward Pass
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 5. Post-processing
    // Reshape matrix: YOLOv8+ outputs [1, 84, 8400] (84 rows: 4 box coords + 80 classes)
    cv::Mat output = outputs[0];
    if (output.dims == 3) {
        output = cv::Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>());
    }
    cv::transpose(output, output); // Transpose to get [8400, 84]

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    float x_factor = frame.cols / INPUT_WIDTH;
    float y_factor = frame.rows / INPUT_HEIGHT;

    for (int i = 0; i < output.rows; ++i) {
        cv::Mat row = output.row(i);
        cv::Mat classes_scores = row.colRange(4, output.cols);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &classIdPoint);

        if (maxClassScore > CONFIDENCE_THRESHOLD) {
            confidences.push_back(maxClassScore);
            classIds.push_back(classIdPoint.x);

            float cx = row.at<float>(0, 0);
            float cy = row.at<float>(0, 1);
            float w = row.at<float>(0, 2);
            float h = row.at<float>(0, 3);

            int left = static_cast<int>((cx - 0.5 * w) * x_factor);
            int top = static_cast<int>((cy - 0.5 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // NMS Filtering to drop overlapping boxes
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    // 6. Draw Bounding Boxes
    for (int idx : nms_indices) {
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID: " + std::to_string(classIds[idx]) + " " + cv::format("%.2f", confidences[idx]);
        cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // Save final visualization

    string output_fn = frame_name.substr(0,frame_name.size()-3) + "output.jpg"; 
    cv::imwrite(output_fn, frame);
    std::cout << "Inference completed successfully. Output saved to output.jpg\n";
    return 0;
}



void readFiles(const string& path, vector<string>& filelist){


    try {
        if (fs::exists(path) && fs::is_directory(path)) {
            for (const auto& entry : fs::directory_iterator(path)) {
                // entry.path() returns the full path layout


                string filename = entry.path().string();
                std::cout << filename << "\n"; 
                filelist.push_back(filename);
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

}


int main(int argc, char**argv){

    if(argc<2){
        std::cerr<<"Usage: " << argv[0] << " <path_to_image>\n";
        return -1;
    }

    //Assign image path from CLI argument
    std::string imagePath = argv[1];

    vector<string> filelist;
    readFiles(imagePath, filelist);

    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov8n.onnx");


    for(string fn: filelist){
        cv::Mat frame = cv::imread(fn);
        detection(net, frame, fn);
    }




}



// g++ -O3 main.cpp -o yolo_infer `pkg-config --cflags --libs opencv4`