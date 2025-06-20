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

struct BatchUnit {
    ov::Tensor input_tensor;
    std::vector<int> indices;
};

struct rec_result {
    std::string text;
    float score;
};

class PPOCRDetector {
private:
    cv::Mat *origin_image;
    std::shared_ptr<ov::Model>det_model;
    ov::InferRequest *request;

    const float det_threshold = 0.3f;
    const float score_threshold = 0.7f;
    const float unclip_ratio = 2.0f;

    cv::Mat inference();

    std::vector<DetectionBox> postProcess(const cv::Mat &prediction);

    float calculateBoxScore(const cv::Mat &prediction, const std::vector<cv::Point> &contour);

    std::vector<cv::Point2f> unclipBox(cv::Point2f *vertices, size_t num_vertices);

    static void sortVerticesClockwise(std::vector<cv::Point2f> &vertices);

    static bool isValidBox(const std::vector<cv::Point2f> &points);

    static void drawResults(cv::Mat &image, const std::vector<DetectionBox> &boxes);

public:
    PPOCRDetector(cv::Mat *image,ov::InferRequest *infer_request) {
        origin_image = image;
        request = infer_request;
    }

    std::vector<DetectionBox> detect();
};


class PPOCRRecognizer {
private:
    int text_num;
    const int batch_num = 6;

    std::vector<DetectionBox> *boxes;
    cv::Mat *origin_image;
    ov::CompiledModel *model;

    std::vector<cv::Mat> text_images;
    std::vector<std::string> ctc_dict;

public:
    PPOCRRecognizer(cv::Mat *origin_image, std::vector<DetectionBox> *det_resluts,
                    ov::CompiledModel *compiled_model);

    void extractSubimage();

    cv::Mat resize_img(const cv::Mat &img, float max_wh_ratio);

    std::vector<BatchUnit> prepare_batches();

    std::vector<ov::InferRequest> run_batches_async(const std::vector<BatchUnit> &batches);

    std::vector<std::string> load_ctc_dict(const std::string &path);

    std::vector<rec_result> decode_ctc_batch(const float *data, const ov::Shape &shape,
                                             const std::vector<std::string> &ctc_dict);

    std::vector<rec_result> recognize();
};


#endif //INFER_H
