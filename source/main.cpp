#include <fstream>
#include <shlwapi.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <windows.h>
#include <main.h>
#include "infer.h"
#include "interface.h"
#include "INIReader.h"


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    INIReader reader("config.ini");

    const std::string det_device = reader.Get("Device", "det_device", "CPU");
    const std::string rec_device = reader.Get("Device", "rec_device", "CPU");
    const std::string det_model_path = reader.Get("Model_Paths", "det_model", "../models/det_model.onnx");
    const std::string rec_model_path = reader.Get("Model_Paths", "rec_model", "../models/rec_model.onnx");

    try {
        ScreenCapture capture;
        capture.StartCapture();
        RECT rect = capture.GetSelectedRect();
        cv::Mat origin_image = capture.CaptureScreenRegion(rect);

        ov::Core core;

        auto det_model = core.read_model(det_model_path);
        auto rec_model = core.read_model(rec_model_path);
        for (const auto &input_layer: rec_model->inputs()) {
            auto input_shape = input_layer.get_partial_shape();
            input_shape[3] = -1;
            rec_model->reshape(input_shape);
        };

        auto ppp_det = init_Det_Model(det_model, origin_image.rows, origin_image.cols);
        auto det_compiled_model = core.compile_model(ppp_det, det_device);
        auto det_infer_request = det_compiled_model.create_infer_request();

        auto ppp_rec = init_Rec_Model(rec_model);
        auto rec_compiled_model = core.compile_model(ppp_rec, rec_device);

        PPOCRDetector detector(&origin_image, &det_infer_request);        auto boxes = detector.detect();

        PPOCRRecognizer Recognizer(&origin_image, &boxes, &rec_compiled_model);
        auto texts = Recognizer.recognize();

        show_result(texts);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
