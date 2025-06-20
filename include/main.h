//
// Created by 10633 on 2025/6/20.
//

#ifndef MAIN_H
#define MAIN_H

std::shared_ptr<ov::Model> init_Det_Model(std::shared_ptr<ov::Model> model, int rows, int cols);

std::shared_ptr<ov::Model> init_Rec_Model(std::shared_ptr<ov::Model> model);


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

#endif //MAIN_H
