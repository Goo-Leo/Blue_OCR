//
// Created by 10633 on 2025/6/12.
//
#include "infer.h"

#include <fstream>
#include <iostream>


// ====================================================
//                      Detector
// ====================================================

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
        subject.emplace_back(vertices[i].x, vertices[i].y);
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
            temp_points.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }

        if (temp_points.size() >= 4) {
            cv::RotatedRect expanded_rect = cv::minAreaRect(temp_points);
            cv::Point2f rect_vertices[4];
            expanded_rect.points(rect_vertices);

            for (auto rect_vertice: rect_vertices) {
                result.push_back(rect_vertice);
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

// ====================================================
//                      Recognizer
// ====================================================

/** Constructor of Recognizer
 * @param origin_image original image opencv mat pointer
 * @param det_resluts detect box pointer
 * @param compiled_model openvino CompiledModel pointer
 */
PPOCRRecognizer::PPOCRRecognizer(cv::Mat *origin_image, std::vector<DetectionBox> *det_resluts,
                                 ov::CompiledModel *compiled_model) {
    this->origin_image = origin_image;
    boxes = det_resluts;
    model = compiled_model;
    text_num = boxes->size();
    ctc_dict = load_ctc_dict("fonts/ppocrv5_dict.txt");
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

/**
 * @brief Extract text images via detect boxes
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

/**
 *  @brief Resize text subimages to the fixed height
 *  @return A Batch of pre-processed data
 */
std::vector<BatchUnit> PPOCRRecognizer::prepare_batches() {
    std::vector<BatchUnit> result;

    int total = static_cast<int>(text_images.size());
    for (int beg = 0; beg < total; beg += batch_num) {
        int end = std::min(beg + batch_num, total);
        int actual_batch = end - beg;

        float max_wh_ratio = 0.0f;
        for (int i = beg; i < end; ++i) {
            float wh_ratio = static_cast<float>(text_images[i].cols) / text_images[i].rows;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        // Resize to the fixed height
        std::vector<cv::Mat> resized;
        std::vector<int> indices;
        const int imgH = 48;
        int imgW = static_cast<int>(32 * max_wh_ratio);
        imgW = ((imgW + 31) / 32) * 32;
        for (int i = beg; i < end; ++i) {
            cv::Mat tmp;
            cv::resize(text_images[i], tmp, cv::Size(imgW, imgH));
            resized.emplace_back(tmp);
            indices.emplace_back(i);
        }

        // batch tensor
        ov::Shape shape = {static_cast<size_t>(actual_batch), static_cast<size_t>(imgH), static_cast<size_t>(imgW), 3};
        ov::Tensor tensor(ov::element::u8, shape);
        size_t img_size = imgH * imgW * 3;
        uint8_t *ptr = tensor.data<uint8_t>();

        for (int i = 0; i < actual_batch; ++i) {
            std::memcpy(ptr + i * img_size, resized[i].data, img_size * sizeof(uint8_t));
        }

        result.push_back({tensor, indices}); // Include indices for result mapping
    }

    return result;
}

/**
 *  @brief Run inference by batches,
 *  @return Request pool for async inference
 */
std::vector<ov::InferRequest> PPOCRRecognizer::run_batches_async(const std::vector<BatchUnit> &batches) {
    std::vector<ov::InferRequest> requests;
    requests.reserve(batches.size());

    for (const auto &batch: batches) {
        ov::InferRequest request = model->create_infer_request();
        request.set_input_tensor(batch.input_tensor);
        request.start_async();
        requests.push_back(std::move(request));
    }

    return requests;
}

std::vector<std::string> PPOCRRecognizer::load_ctc_dict(const std::string &path) {
    std::ifstream file(path);
    std::vector<std::string> dict;

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CTC dictionary file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove potential carriage return and newline characters
        if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict.push_back(line);
    }

    // Insert blank token at the beginning for CTC
    dict.insert(dict.begin(), "blank");
    return dict;
}

std::vector<rec_result> PPOCRRecognizer::decode_ctc_batch(
    const float* data,
    const ov::Shape& shape,  // [N, T, C]
    const std::vector<std::string>& ctc_dict)
{
    std::vector<rec_result> results;
    int N = static_cast<int>(shape[0]);
    int T = static_cast<int>(shape[1]);
    int C = static_cast<int>(shape[2]);

    results.reserve(N);

#pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        std::string text;
        float conf_sum = 0.0f;
        int char_count = 0;
        int last_index = -1;

        for (int t = 0; t < T; ++t) {
            const float* prob = data + (n * T + t) * C;

            int max_index = 0;
            float max_prob = prob[0];
            for (int c = 1; c < C; ++c) {
                if (prob[c] > max_prob) {
                    max_prob = prob[c];
                    max_index = c;
                }
            }

            if (max_index == 0 || max_index == last_index) {
                continue;  // blank or repeated
            }

            if (max_index < ctc_dict.size()) {
#pragma omp critical
                text += ctc_dict[max_index];
                conf_sum += max_prob;
                char_count += 1;
            }

            last_index = max_index;
        }

        float avg_score = char_count > 0 ? conf_sum / char_count : 0.0f;

#pragma omp critical
        results.emplace_back(rec_result{text, avg_score});
    }

    return results;
}


/**
 * @brief Recognize Progress
 * @return Recognize results: texts and confidences
 */
std::vector<rec_result> PPOCRRecognizer::recognize() {
    std::vector<rec_result> final_results(text_num); // Pre-allocate with correct size

    extractSubimage();

    auto batches = prepare_batches();
    auto start_time = std::chrono::high_resolution_clock::now();

    auto async_requests = run_batches_async(batches);

    // Process results from all batches
    for (size_t batch_idx = 0; batch_idx < async_requests.size(); ++batch_idx) {
        async_requests[batch_idx].wait();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto out_tensor = async_requests[batch_idx].get_output_tensor();
        const float *output_data = out_tensor.data<float>();
        auto shape = out_tensor.get_shape(); // [N, T, C]

        std::cout << "Inference time: "<< duration << std::endl;
        // Decode current batch
        auto batch_results = decode_ctc_batch(output_data, shape, ctc_dict);

        // Map batch results back to original positions using indices
        const auto &batch_indices = batches[batch_idx].indices;
        for (size_t i = 0; i < batch_results.size() && i < batch_indices.size(); ++i) {
            int original_idx = batch_indices[i];
            if (original_idx < static_cast<int>(final_results.size())) {
                final_results[original_idx] = batch_results[i];
            }
        }
    }

    return final_results;
}
