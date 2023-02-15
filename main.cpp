#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <chrono>

#include "yolov5.h"
// #include "yolov5-uint8.h"


int main()
{
    YOLOv5Detector detector;
    // detector.init("./models/yolov5lv5opt.tmfile", 640, 640, 4, 0.25, 0.45);
    detector.init("/home/Park/yolo_Tengine/models/yolov5m6v5opt.tmfile", 1280, 1280, 4, 0.25, 0.45);
    // detector.init("/home/Park/yolo_Tengine/models/yolov5m6v5-uint8.tmfile", 1280, 1280, 4, 0.25, 0.45);
    cv::Mat frame;
    
    frame = cv::imread("/home/Park/yolo_Tengine/images/1.jpg");
    auto t1 = std::chrono::high_resolution_clock::now();
    detector.infer(frame);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout<<"t2 - t1: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<std::endl;

    frame = cv::imread("/home/Park/yolo_Tengine/images/dog.jpg");
    auto t3 = std::chrono::high_resolution_clock::now();
    detector.infer(frame);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout<<"t4 - t3: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()<<std::endl;

    frame = cv::imread("/home/Park/yolo_Tengine/images/20221202174912_3-1.jpg");
    auto t5 = std::chrono::high_resolution_clock::now();
    detector.infer(frame);
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout<<"t6 - t5: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()<<std::endl;

    frame = cv::imread("/home/Park/yolo_Tengine/images/current.jpg");
    auto t7 = std::chrono::high_resolution_clock::now();
    detector.infer(frame);
    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout<<"t8 - t7: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count()<<std::endl;

    return 1;
}