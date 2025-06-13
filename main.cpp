#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "infer.h"
#include "interface.h"

std::shared_ptr<ov::Model> init_Det_Model(std::shared_ptr<ov::Model> model, int rows, int cols);
void get_Srceen_shot();

int main() {
    get_Srceen_shot();
    try {
        cv::Mat origin_image = cv::imread("screenshot.png", cv::IMREAD_COLOR);
        origin_image.convertTo(origin_image, CV_8UC1);

        const std::string output_path = "result.jpg";
        const std::string device = "NPU";
        const std::string det_model_path = "det_model.onnx";
        const std::string rec_model_path = "rec_model.onnx";

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

void get_Srceen_shot() {
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    std::wcout << L"屏幕截图工具" << std::endl;
    std::wcout << L"使用方法：" << std::endl;
    std::wcout << L"1. 按下鼠标左键并拖拽选择截图区域" << std::endl;
    std::wcout << L"2. 释放鼠标完成选择" << std::endl;
    std::wcout << L"3. 按ESC键取消截图" << std::endl;
    std::wcout << L"按任意键开始截图..." << std::endl;

    system("pause");

    ScreenCapture capture;

    if (capture.StartCapture()) {
        RECT rect = capture.GetSelectedRect();

        if (rect.right > rect.left && rect.bottom > rect.top) {
            std::wstring filename = L"screenshot.png";

            if (capture.CaptureScreenRegion(rect, filename)) {
                std::wcout << L"截图已保存为: " << filename << std::endl;
                std::wcout << L"区域: (" << rect.left << ", " << rect.top
                          << ") - (" << rect.right << ", " << rect.bottom << ")" << std::endl;
            } else {
                std::wcout << L"保存截图失败!" << std::endl;
            }
        } else {
            std::wcout << L"未选择有效区域" << std::endl;
        }
    } else {
        std::wcout << L"创建截图界面失败!" << std::endl;
    }

    // 清理GDI+
    GdiplusShutdown(gdiplusToken);
}