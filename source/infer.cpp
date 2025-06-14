//
// Created by 10633 on 2025/6/12.
//
#include "../include/infer.h"
#include <iostream>


cv::Mat PPOCRDetector::inference() {
    try {
        ov::Tensor input_tensor = ov::Tensor(ov::element::u8,
                                             {
                                                 1,
                                                 static_cast<size_t>(origin_image->rows),
                                                 static_cast<size_t>(origin_image->cols),
                                                 3
                                             });
        std::memcpy(input_tensor.data<uint8_t>(), origin_image->data,
                    origin_image->total() * origin_image->elemSize());
        // free

        request->set_input_tensor(input_tensor);
        std::cout << "input shape: " << input_tensor.get_shape() << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        request->infer();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

        auto output_tensor = request->get_output_tensor();
        auto shape = output_tensor.get_shape();

        std::cout << "Output shape: ";
        for (auto dim: shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        float *output_data = output_tensor.data<float>();
        cv::Mat output_mat(static_cast<int>(shape[2]), static_cast<int>(shape[3]), CV_32F, output_data);

        return output_mat;
    } catch (const std::exception &e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        throw;
    }
}


std::vector<DetectionBox> PPOCRDetector::postProcess(const cv::Mat &prediction) {
    std::vector<DetectionBox> boxes;
    try {
        // get binary mask and find contours
        cv::Mat binary_mask;
        cv::threshold(prediction, binary_mask, det_threshold, 1.0, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U, 255);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(binary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        std::cout << "Found " << contours.size() << " contours" << std::endl;

        // calculate boxes
        float scale_x = static_cast<float>(origin_image->cols) / static_cast<float>(prediction.cols);
        float scale_y = static_cast<float>(origin_image->rows) / static_cast<float>(prediction.rows);

        // Process contours with better filtering
        for (const auto &contour: contours) {
            if (contour.size() < 4 || cv::contourArea(contour) < 9) {
                continue;
            }

            cv::RotatedRect rect = cv::minAreaRect(contour);
            if (rect.size.area() < 16) {
                continue;
            }

            float score = calculateBoxScore(prediction, contour);
            if (score < score_threshold || std::isnan(score)) {
                continue;
            }

            cv::Point2f vertices[4];
            rect.points(vertices);
            // std::vector<cv::Point2f> sorted_vertices(vertices, vertices + 4);
            // sortVerticesClockwise(sorted_vertices);

            std::vector<cv::Point2f> expanded_box = unclipBox(vertices, 4);
            if (expanded_box.empty() || expanded_box.size() != 4) {
                expanded_box.assign(vertices, vertices + 4);
            }

            DetectionBox det_box;
            det_box.score = score;
            for (const auto &point: expanded_box) {
                det_box.points.emplace_back(
                    std::clamp(point.x * scale_x, 0.0f, static_cast<float>(origin_image->cols - 1)),
                    std::clamp(point.y * scale_y, 0.0f, static_cast<float>(origin_image->rows - 1))
                );
            }
            // std::vector<cv::Point2f> sorted_vertices(det_box.points[0], vertices + 4);
            sortVerticesClockwise(det_box.points);

            // Validate box geometry
            if (isValidBox(det_box.points)) {
                boxes.push_back(det_box);
            }
        }

        std::cout << "Detected " << boxes.size() << " text regions" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Post-processing error: " << e.what() << std::endl;
    }

    return boxes;
}

float PPOCRDetector::calculateBoxScore(const cv::Mat &prediction, const std::vector<cv::Point> &contour) {
    cv::Mat mask = cv::Mat::zeros(prediction.size(), CV_8U);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point> >{contour}, cv::Scalar(255));

    cv::Scalar mean_score = cv::mean(prediction, mask);
    return static_cast<float>(mean_score[0]);
}

std::vector<cv::Point2f> PPOCRDetector::unclipBox(cv::Point2f *vertices, size_t num_vertices = 4) {
    std::vector<cv::Point2f> result;

    Clipper2Lib::PathD subject;
    for (size_t i = 0; i < num_vertices; ++i) {
        subject.push_back(Clipper2Lib::PointD(vertices[i].x, vertices[i].y));
    }

    double area = std::abs(Clipper2Lib::Area(subject));
    double perimeter = 0.0;
    for (size_t i = 0; i < num_vertices; ++i) {
        int next = (i + 1) % num_vertices;
        double dx = subject[next].x - subject[i].x;
        double dy = subject[next].y - subject[i].y;
        perimeter += std::sqrt(dx * dx + dy * dy);
    }

    double distance = area * unclip_ratio / perimeter;

    Clipper2Lib::PathsD solution;
    solution = Clipper2Lib::InflatePaths({subject}, distance,
                                         Clipper2Lib::JoinType::Round,
                                         Clipper2Lib::EndType::Polygon);

    if (!solution.empty() && !solution[0].empty()) {
        std::vector<cv::Point2f> temp_points;
        for (const auto &pt: solution[0]) {
            temp_points.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        }

        if (temp_points.size() >= 4) {
            cv::RotatedRect expanded_rect = cv::minAreaRect(temp_points);
            cv::Point2f rect_vertices[4];
            expanded_rect.points(rect_vertices);

            for (int i = 0; i < 4; ++i) {
                result.push_back(rect_vertices[i]);
            }
        } else {
            for (size_t i = 0; i < num_vertices; ++i) {
                result.push_back(vertices[i]);
            }
        }
    }

    return result;
}


void PPOCRDetector::sortVerticesClockwise(std::vector<cv::Point2f> &vertices) {
    if (vertices.size() != 4) return;

    cv::Point2f center(0, 0);
    for (const auto &pt: vertices) {
        center += pt;
    }
    center *= 0.25f;

    std::sort(vertices.begin(), vertices.end(),
              [center](const cv::Point2f &a, const cv::Point2f &b) {
                  double angle_a = std::atan2(a.y - center.y, a.x - center.x);
                  double angle_b = std::atan2(b.y - center.y, b.x - center.x);
                  return angle_a < angle_b;
              });
}


bool PPOCRDetector::isValidBox(const std::vector<cv::Point2f> &points) {
    if (points.size() < 4) return false;

    float width1 = cv::norm(points[0] - points[1]);
    float width2 = cv::norm(points[2] - points[3]);
    float height1 = cv::norm(points[0] - points[3]);
    float height2 = cv::norm(points[1] - points[2]);

    float min_width = std::min(width1, width2);
    float min_height = std::min(height1, height2);

    return min_width > 3 && min_height > 3;
}

void PPOCRDetector::drawResults(cv::Mat &image, const std::vector<DetectionBox> &boxes) {
    for (const auto &box: boxes) {
        if (box.points.size() >= 4) {
            std::vector<cv::Point> int_points;
            for (const auto &pt: box.points) {
                int_points.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
            }

            cv::polylines(image, std::vector<std::vector<cv::Point> >{int_points},
                          true, cv::Scalar(0, 255, 0), 1);

            std::string score_text = std::to_string(box.score);
            score_text = score_text.substr(0, 4);
            cv::putText(image, score_text,
                        int_points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 0, 0), 1);
        }
    }
}

std::vector<DetectionBox> PPOCRDetector::detect(const std::string &output_path = "result.jpg") {
    std::cout << "Image size: " << origin_image->size() << std::endl;

    cv::Mat prediction = inference();

    std::vector<DetectionBox> boxes = postProcess(prediction);

    cv::Mat result_image = origin_image->clone();
    drawResults(result_image, boxes);

    cv::imwrite(output_path, result_image);
    std::cout << "Result saved to: " << output_path << std::endl;

    return boxes;
}


PPOCRRecognizer::PPOCRRecognizer(cv::Mat *origin_image, std::vector<DetectionBox> *det_resluts,
                                 ov::InferRequest *infer_request) {
    this->origin_image = origin_image;
    boxes = det_resluts;
    request = infer_request;
    text_num = boxes->size();

    std::sort(boxes->begin(), boxes->end(),
              [](const DetectionBox &a, const DetectionBox &b) {
                  return std::tie(a.points[0].y, a.points[0].x) < std::tie(b.points[0].y, b.points[0].x);
              });

    if (det_resluts->size() > 1) {
        for (size_t i = 0; i < text_num - 1; ++i) {
            if (std::abs((*boxes)[i + 1].points[0].y - (*boxes)[i].points[0].y) < 10.0f &&
                ((*boxes)[i + 1].points[0].x < (*boxes)[i].points[0].x)) {
                std::swap((*boxes)[i], (*boxes)[i + 1]);
            }
        }
    }
}

/** extract text images for reco input
 *
 */
void PPOCRRecognizer::extractSubimage() {
    for (int i = 0; i < text_num; ++i) {
        if ((*boxes)[i].points.size() != 4) {
            throw std::runtime_error("Exactly 4 points required");
        }

        const float width = cv::norm((*boxes)[i].points[1] - (*boxes)[i].points[0]);
        const float height = cv::norm((*boxes)[i].points[2] - (*boxes)[i].points[1]);

        std::vector<cv::Point2f> dstPoints = {
            {0, 0},
            {width, 0},
            {width, height},
            {0, height}
        };

        cv::Mat transform = cv::getPerspectiveTransform((*boxes)[i].points, dstPoints);

        cv::Mat dst;
        cv::warpPerspective(
            *origin_image, dst, transform,
            cv::Size(static_cast<int>(width), static_cast<int>(height)),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT
        );

        text_images.emplace_back(dst);
    }
}

cv::Mat PPOCRRecognizer::normalize(const cv::Mat &input_img, float max_wh_ratio) {
    const int imgC = 3;
    const int imgH = 48;
    int imgW = static_cast<int>(32 * max_wh_ratio);

    float h = input_img.rows;
    float w = input_img.cols;
    float ratio = w / h;
    int resized_w = (std::ceil(imgH * ratio) > imgW) ? imgW : static_cast<int>(std::ceil(imgH * ratio));

    cv::Mat resized_img;
    cv::resize(input_img, resized_img, cv::Size(resized_w, imgH), 0, 0, cv::INTER_LINEAR);

    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F);
    float_img = float_img / 255.0;
    float_img = (float_img - 0.5) / 0.5;

    cv::Mat padded_img = cv::Mat::zeros(imgH, imgW, CV_32FC3);
    cv::Rect roi(0, 0, resized_w, imgH);
    float_img.copyTo(padded_img(roi));

    std::vector<cv::Mat> channels;
    cv::split(padded_img, channels);

    cv::Mat chw_img;
    cv::vconcat(channels, chw_img);

    return chw_img;
}

std::vector<cv::Mat> PPOCRRecognizer::batch_process_images(
    const std::vector<cv::Mat>& img_crop_list,
    const std::vector<int>& indices,
    int start_index) {

    int end_index = std::min(static_cast<int>(img_crop_list.size()), start_index + batch_num);

    float max_wh_ratio = 0.0f;
    for (int i = start_index; i < end_index; i++) {
        int idx = indices[i];
        float h = img_crop_list[idx].rows;
        float w = img_crop_list[idx].cols;
        float wh_ratio = w / h;
        if (wh_ratio > max_wh_ratio) {
            max_wh_ratio = wh_ratio;
        }
    }

    std::vector<cv::Mat> batch;
    for (int i = start_index; i < end_index; i++) {
        int idx = indices[i];
        cv::Mat processed = normalize(img_crop_list[idx], max_wh_ratio);
        batch.push_back(processed);
    }

    return batch;
}

std::vector<rec_result> PPOCRRecognizer::process_text_regions(
    const std::vector<cv::Mat> &img_crop_list,
    int batch_size = 6) {
    // 1. 计算所有图像的宽高比并排序
    std::vector<std::pair<float, int>> wh_ratios;
    for (int i = 0; i < img_crop_list.size(); i++) {
        float h = img_crop_list[i].rows;
        float w = img_crop_list[i].cols;
        wh_ratios.emplace_back(w / h, i);
    }

    std::sort(wh_ratios.begin(), wh_ratios.end());

    std::vector<int> indices;
    for (const auto& pair : wh_ratios) {
        indices.push_back(pair.second);
    }

    for (int beg_img_no = 0; beg_img_no < img_crop_list.size(); beg_img_no += batch_size) {
        std::vector<cv::Mat> batch = batch_process_images(
            img_crop_list, indices, beg_img_no);


        int batch_count = batch.size();
        int channels = 3;
        int height = 48;
        int width = batch[0].cols; // 所有图像宽度相同

        // 创建连续内存存储批处理数据
        batch_tensor.(batch_count * channels * height, width, CV_32F);

        // 将每个处理后的图像复制到批处理张量中
        for (int i = 0; i < batch_count; i++) {
            cv::Mat roi = batch_tensor.rowRange(
                i * channels * height,
                (i + 1) * channels * height);
            batch[i].copyTo(roi);
        }
    }
}

/**
 *
 * @return recognize results: texts and scores
 */
std::vector<rec_result> PPOCRRecognizer::recognize() {
    std::vector<rec_result> results;

    extractSubimage();

    process_text_regions(text_images);

    return results;
}
