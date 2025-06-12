//
// Created by 10633 on 2025/6/3.
//
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <clipper2/clipper.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#ifndef INFER_H
#define INFER_H

struct DetectionBox {
    std::vector<cv::Point2f> points;
    float score;
};

struct rec_result {
    std::string text;
    float score;
};

class PPOCRDetector {
private:
    cv::Mat *origin_image;
    ov::InferRequest *request;

    const float det_threshold = 0.3f;
    const float score_threshold = 0.7f;
    const float unclip_ratio = 2.0f;

    cv::Mat inference();

    std::vector<DetectionBox> postProcess(const cv::Mat &prediction);

    float calculateBoxScore(const cv::Mat &prediction, const std::vector<cv::Point> &contour);

    std::vector<cv::Point2f> unclipBox(cv::Point2f *vertices, size_t num_vertices);

    void sortVerticesClockwise(std::vector<cv::Point2f> &vertices);

    bool isValidBox(const std::vector<cv::Point2f> &points);

    void drawResults(cv::Mat &image, const std::vector<DetectionBox> &boxes);

public:
    PPOCRDetector(cv::Mat *image, ov::InferRequest *infer_request) {
        origin_image = image;
        request = infer_request;
    }

    std::vector<DetectionBox> detect(const std::string &output_path);
};


class PPOCRRecognizer {
private:
    int text_num;
    const int batch_num = 6;
    ov::InferRequest *request;
    cv::Mat *origin_image;
    std::vector<cv::Mat> text_images;
    std::vector<DetectionBox> boxes;

public:
    PPOCRRecognizer(cv::Mat *origin_image, std::vector<DetectionBox> det_resluts,
                    ov::InferRequest *infer_request);

    std::vector<rec_result> Recognizer();
};


#endif //INFER_H
