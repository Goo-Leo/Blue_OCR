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
            std::vector<cv::Point2f> sorted_vertices(vertices, vertices + 4);
            sortVerticesClockwise(sorted_vertices);

            std::vector<cv::Point2f> expanded_box = unclipBox(sorted_vertices.data(), 4);
            if (expanded_box.empty() || expanded_box.size() != 4) {
                expanded_box.assign(sorted_vertices.begin(), sorted_vertices.end());
            }

            DetectionBox det_box;
            det_box.score = score;
            for (const auto &point: expanded_box) {
                det_box.points.push_back(cv::Point2f(
                    std::clamp(point.x * scale_x, 0.0f, static_cast<float>(origin_image->cols - 1)),
                    std::clamp(point.y * scale_y, 0.0f, static_cast<float>(origin_image->rows - 1))
                ));
            }

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
                int_points.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }

            // 绘制多边形
            cv::polylines(image, std::vector<std::vector<cv::Point> >{int_points},
                          true, cv::Scalar(0, 255, 0), 2);

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


PPOCRRecognizer::PPOCRRecognizer(cv::Mat *origin_image, std::vector<DetectionBox> det_resluts,
                                 ov::InferRequest *infer_request) {
    boxes = det_resluts;
    request = infer_request;
    text_num = boxes.size();
    std::sort(boxes.begin(), boxes.end(),
              [](const DetectionBox &a, const DetectionBox &b) {
                  return std::tie(a.points[0].y, a.points[0].x) < std::tie(b.points[0].y, b.points[0].x);
              });

    if (boxes.size() > 1) {
        for (size_t i = 0; i < text_num - 1; ++i) {
            if (std::abs(boxes[i + 1].points[0].y - boxes[i].points[0].y) < 10.0f &&
                (boxes[i + 1].points[0].x < boxes[i].points[0].x)) {
                std::swap(boxes[i], boxes[i + 1]);
            }
        }
    }
}

// std::vector<rec_result> PPOCRRecognizer::PPOCRRecognizer::Recognizer() {
//     for (int i = 0; i < text_num; ++i) {
//         cv::Rect bounding_box = cv::boundingRect(boxes[i].points);
//         // text_images.push_back();
//     }
//
//
//     request->set_callback([&](std::exception_ptr ex_ptr) {
//         if (!ex_ptr) {
//             // all done. Output data can be processed.
//             // You can fill the input data and run inference one more time:
//             request->start_async();
//         } else {
//             // Something wrong, you can analyze exception_ptr
//         }
//     });
// }
