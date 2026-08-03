#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Initialize VideoCapture with the default camera index (0)
    cv::VideoCapture cap(0);

    // 2. Safety Check: Verify if the webcam opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the webcam." << std::endl;
        return -1;
    }

    // Allocate a matrix to hold individual video frames
    cv::Mat frame;
    const std::string windowName = "Webcam Live Feed";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    std::cout << "Streaming live video. Press 'ESC' to exit..." << std::endl;

    int count = 0;

    // 3. Continuous frame capture loop
    while (true) {
        // Grab and decode the latest frame
        cap >> frame;

        // Verify the frame is not empty
        if (frame.empty()) {
            std::cerr << "Error: Captured an empty frame." << std::endl;
            break;
        }
        count ++;

        // 4. Display the frame in the created window
        cv::imshow(windowName, frame);

        if (count%30==0){
            char filename[256];
            sprintf(filename, "./images/webcam_%d.jpg", count);
            cv::imwrite(filename, frame);
        }
        // 5. Wait for 30ms and check if 'ESC' (ASCII 27) is pressed
        char key = static_cast<char>(cv::waitKey(30));
        if (key == 27) {
            break;
        }
    }

    // 6. Free hardware resources and clean up windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}


// g++ main.cpp -o webcam_stream `pkg-config --cflags --libs opencv4`