#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "include/infer.h"

std::shared_ptr<ov::Model> init_Det_Model(std::shared_ptr<ov::Model> model, int rows, int cols);

int main() {
    try {
        cv::Mat origin_image = cv::imread("img_2.png", cv::IMREAD_COLOR);
        origin_image.convertTo(origin_image, CV_8UC1);

        const std::string output_path = "result.jpg";
        const std::string device = "NPU";
        const std::string det_model_path = "./det_model.onnx";
        const std::string rec_model_path = "./rec_model.onnx";

        ov::Core core;

        auto det_model = core.read_model(det_model_path);
        auto rec_model = core.read_model(rec_model_path);

        std::shared_ptr<ov::Model> ppp_model = init_Det_Model(det_model, origin_image.rows, origin_image.cols);
        auto det_compiled_model = core.compile_model(ppp_model, device);

        for (auto input_layer: rec_model->inputs()) {
            auto input_shape = input_layer.get_partial_shape();
            input_shape[3] = -1;
            rec_model->reshape(input_shape);
        };
        auto rec_compiled_model = core.compile_model(rec_model, "AUTO");


        ov::InferRequest infer_request_det = det_compiled_model.create_infer_request();
        PPOCRDetector detector(&origin_image, &infer_request_det);
        auto boxes = detector.detect(output_path);

        ov::InferRequest infer_request_rec = rec_compiled_model.create_infer_request();
        PPOCRRecognizer recognizer(&origin_image,boxes, &infer_request_rec);

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


std::shared_ptr<ov::Model> init_Det_Model(std::shared_ptr<ov::Model> model, int rows, int cols) {
    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

    ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR,
                    static_cast<size_t>(640), static_cast<size_t>(640))
            .mean({123.675f, 116.28f, 103.53f})
            .scale({58.395f, 57.12f, 57.375f})
            .convert_layout("NCHW");
    model = ppp.build();

    model->reshape({1, rows, cols, 3});

    return model;
}
