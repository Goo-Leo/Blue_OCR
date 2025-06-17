#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "infer.h"
#include "interface.h"

std::shared_ptr<ov::Model> init_Det_Model(std::shared_ptr<ov::Model> model, int rows, int cols);

std::shared_ptr<ov::Model> init_Rec_Model(std::shared_ptr<ov::Model> model);

int main() {
    ScreenCapture capture;
    try {
        capture.StartCapture();
        RECT rect = capture.GetSelectedRect();
        cv::Mat origin_image = capture.CaptureScreenRegion(rect);
        // cv::Mat origin_image = cv::imread("img.png");
        origin_image.convertTo(origin_image, CV_8UC1);

        const std::string output_path = "result.jpg";
        const std::string device = "NPU";
        const std::string det_model_path = "det_model.onnx";
        const std::string rec_model_path = "rec_model.onnx";

        ov::Core core;

        auto det_model = core.read_model(det_model_path);
        auto rec_model = core.read_model(rec_model_path);
        for (const auto &input_layer: rec_model->inputs()) {
            auto input_shape = input_layer.get_partial_shape();
            input_shape[3] = -1;
            rec_model->reshape(input_shape);
        };

        auto ppp_det = init_Det_Model(det_model, origin_image.rows, origin_image.cols);
        auto det_compiled_model = core.compile_model(ppp_det, device);
        auto det_infer_request = det_compiled_model.create_infer_request();

        auto ppp_rec = init_Rec_Model(rec_model);
        auto rec_compiled_model = core.compile_model(ppp_rec, "GPU");

        PPOCRDetector detector(&origin_image, &det_infer_request);
        auto boxes = detector.detect(output_path);

        PPOCRRecognizer Recognizer(&origin_image, &boxes, &rec_compiled_model);
        auto texts = Recognizer.recognize();

        std::ofstream out("ocr_results.txt", std::ios::out | std::ios::binary);
        out << "\xEF\xBB\xBF";

        for (const auto& text : texts) {
            out << text.text << " (score: " << text.score << ")" << std::endl;
        }

        out.close();

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
            .convert_element_type(ov::element::f16)
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

std::shared_ptr<ov::Model> init_Rec_Model(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);

    ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .mean({127.5f, 127.5f, 127.5f})
            .scale({127.5f, 127.5f, 127.5f})
            .convert_layout("NCHW");

    ppp.input().model().set_layout("NCHW");
    model = ppp.build();

    return model;
}
