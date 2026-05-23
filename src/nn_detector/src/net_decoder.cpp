#include <detector/net_decoder.h>
#include <utils/common.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string dims_to_string(const std::vector<size_t> &dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

}  // namespace

NetDecoderBase::NetDecoderBase(toml::value &config,const rclcpp::Logger& _logger): logger(_logger) {
    INPUT_W = config.at("INPUT_W").as_integer();
    INPUT_H = config.at("INPUT_H").as_integer();

    NUM_CLASSES = config.at("NUM_CLASSES").as_integer();
    NUM_COLORS = config.at("NUM_COLORS").as_integer();

    BBOX_CONF_THRESH = config.at("BBOX_CONF_THRESH").as_floating();
}

YOLOv5Decoder::YOLOv5Decoder(toml::value &config, const rclcpp::Logger& _logger) : NetDecoderBase(config,_logger) {
    anchors = toml::get<decltype(anchors)>(config.at("anchors"));
}

void YOLOv5Decoder::set_layer_info(int layer_index, const std::vector<size_t> &dimensions) {
    assert((int)layers.size() == layer_index && "set_layer_info should be called in order");
    assert(dimensions.size() == 5 && "model should be 5-dim");
    assert(dimensions[2] == dimensions[3] && "model output should be square");
    int stride = INPUT_H / dimensions[2];

    YOLOv5LayerInfo layer = {layer_index,(int)dimensions[1],(int)dimensions[2],(int)dimensions[3],(int)dimensions[4],stride};

    RCLCPP_INFO(logger,"layer %d: num_anchors=%d, out_h=%d, out_w=%d, num_outputs=%d, stride=%d", layer_index, layer.num_anchors, layer.out_h, layer.out_w, layer.num_outputs, layer.stride);
    assert(this->check_num_outputs(layer.num_outputs) && "num_output check failed");

    layers.push_back(layer);
    return;
}

bool YOLOv5Decoder::check_num_outputs(int num_outputs) { return num_outputs == 1 + 4 + 10 + NUM_CLASSES + NUM_COLORS; }

void YOLOv5Decoder::decode(int layer_index, const float *prob, std::vector<Armor> &objects) {
    assert((int)layers.size() > layer_index && "layer_index out of range");
    auto [_, na, out_h, out_w, no, stride] = layers[layer_index];

    std::vector<float> anchor;
    switch (stride) {
        case 8:
            anchor = anchors[0];
            break;
        case 16:
            anchor = anchors[1];
            break;
        case 32:
            anchor = anchors[2];
            break;
        case 4:
            anchor = anchors[3];
            break;
        default:
            assert(false && "Unknown layer stride");
            break;
    }
    std::vector<float> pred_data_v;
    pred_data_v.resize(no);
    float* pred_data = pred_data_v.data();
    // float pred_data[no];
    // [x, y, w, h, conf, (x,y)*5, hot(class), hot(color)]
    for (int a_id = 0; a_id < na; ++a_id) {
        for (int h_id = 0; h_id < out_h; ++h_id) {
            for (int w_id = 0; w_id < out_w; ++w_id) {
                int data_idx = (a_id * out_h * out_w + h_id * out_w + w_id) * no;
                float obj_conf = sigmoid(prob[data_idx + 4]);
                if (obj_conf > BBOX_CONF_THRESH) {
                    // std::cout << obj_conf << std::endl;
                    sigmoid(prob + data_idx, pred_data, 5);
                    sigmoid(prob + data_idx + 15, pred_data + 15, NUM_CLASSES + NUM_COLORS);
                    memcpy(pred_data + 5, prob + data_idx + 5, sizeof(float) * 10);
                    int cls_id = std::distance(pred_data + 15, std::max_element(pred_data + 15, pred_data + 15 + NUM_CLASSES));
                    int col_id = std::distance(pred_data + 15 + NUM_CLASSES, std::max_element(pred_data + 15 + NUM_CLASSES, pred_data + 15 + NUM_CLASSES + NUM_COLORS));

                    double final_conf = obj_conf * sqrt(pred_data[15 + cls_id] * pred_data[15 + NUM_CLASSES + col_id]);
                    if (final_conf > BBOX_CONF_THRESH) {
                        // std::cout << final_conf << " " << col_id << " "
                        //           << cls_id << std::endl;
                        Armor now;
                        float x = (pred_data[0] * 2.0 - 0.5 + w_id) * stride;
                        float y = (pred_data[1] * 2.0 - 0.5 + h_id) * stride;
                        float w = pow(pred_data[2] * 2, 2) * anchor[a_id * 2];
                        float h = pow(pred_data[3] * 2, 2) * anchor[a_id * 2 + 1];

                        for (int p = 0; p < 5; ++p) {
                            float px = (pred_data[5 + p * 2] * anchor[a_id * 2] + w_id * stride);
                            float py = (pred_data[5 + p * 2 + 1] * anchor[a_id * 2 + 1] + h_id * stride);
                            // px = std::max(std::min(px, (float)(INPUT_W)), 0.f);
                            // py = std::max(std::min(py, (float)(INPUT_H)), 0.f);
                            now.pts[p] = cv::Point2f(px, py);
                            // std::cout << px << " " << py  << " ";
                        }
                        // std::cout << std::endl;

                        float x0 = (x - w * 0.5);
                        float y0 = (y - h * 0.5);
                        float x1 = (x + w * 0.5);
                        float y1 = (y + h * 0.5);

                        // x0 = std::max(std::min(x0, (float)(INPUT_W)), 0.f);
                        // y0 = std::max(std::min(y0, (float)(INPUT_H)), 0.f);
                        // x1 = std::max(std::min(x1, (float)(INPUT_W)), 0.f);
                        // y1 = std::max(std::min(y1, (float)(INPUT_H)), 0.f);

                        now.rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);
                        now.conf = final_conf;
                        now.color = col_id;
                        now.type = cls_id;
                        now.size = 0;
                        objects.push_back(now);
                    }
                }
            }
        }
    }
    return;
}

YOLOv5_1_Decoder::YOLOv5_1_Decoder(toml::value &config, const rclcpp::Logger& _logger) : YOLOv5Decoder(config, _logger) {
    NUM_TSIZES = config.at("NUM_TSIZES").as_integer();
}

bool YOLOv5_1_Decoder::check_num_outputs(int num_outputs) { return num_outputs == 1 + 4 + 10 + NUM_CLASSES + NUM_COLORS + NUM_TSIZES; }

void YOLOv5_1_Decoder::decode(int layer_index, const float *prob, std::vector<Armor> &objects) {
    assert((int)layers.size() > layer_index && "layer_index out of range");
    auto [_, na, out_h, out_w, no, stride] = layers[layer_index];

    std::vector<float> anchor;
    switch (stride) {
        case 8:
            anchor = anchors[0];
            break;
        case 16:
            anchor = anchors[1];
            break;
        case 32:
            anchor = anchors[2];
            break;
        case 4:
            anchor = anchors[3];
            break;
        default:
            assert(false && "Unknown layer stride");
            break;
    }
    std::vector<float> pred_data_v;
    pred_data_v.resize(no);
    float* pred_data = pred_data_v.data();
    // [x, y, w, h, conf, (x,y)*5, hot(class), hot(color), hot(tsize)]
    for (int a_id = 0; a_id < na; ++a_id) {
        for (int h_id = 0; h_id < out_h; ++h_id) {
            for (int w_id = 0; w_id < out_w; ++w_id) {
                int data_idx = (a_id * out_h * out_w + h_id * out_w + w_id) * no;
                float obj_conf = sigmoid(prob[data_idx + 4]);
                if (obj_conf > BBOX_CONF_THRESH) {
                    // std::cout << obj_conf << std::endl;
                    sigmoid(prob + data_idx, pred_data, 5);
                    sigmoid(prob + data_idx + 15, pred_data + 15, NUM_CLASSES + NUM_COLORS + NUM_TSIZES);
                    memcpy(pred_data + 5, prob + data_idx + 5, sizeof(float) * 10);
                    int cls_id = std::distance(pred_data + 15, std::max_element(pred_data + 15, pred_data + 15 + NUM_CLASSES));
                    int col_id = std::distance(pred_data + 15 + NUM_CLASSES, std::max_element(pred_data + 15 + NUM_CLASSES, pred_data + 15 + NUM_CLASSES + NUM_COLORS));
                    int ts_id =
                        std::distance(pred_data + 15 + NUM_CLASSES + NUM_COLORS, std::max_element(pred_data + 15 + NUM_CLASSES + NUM_COLORS, pred_data + 15 + NUM_CLASSES + NUM_COLORS + NUM_TSIZES));

                    double final_conf = obj_conf * pow(pred_data[15 + cls_id] * pred_data[15 + NUM_CLASSES + col_id] * pred_data[15 + NUM_CLASSES + NUM_COLORS + ts_id], 1.0 / 3.0);
                    if (final_conf > BBOX_CONF_THRESH) {
                        // std::cout << final_conf << " " << col_id << " "
                        //           << cls_id << std::endl;
                        Armor now;
                        float x = (pred_data[0] * 2.0 - 0.5 + w_id) * stride;
                        float y = (pred_data[1] * 2.0 - 0.5 + h_id) * stride;
                        float w = pow(pred_data[2] * 2, 2) * anchor[a_id * 2];
                        float h = pow(pred_data[3] * 2, 2) * anchor[a_id * 2 + 1];

                        for (int p = 0; p < 5; ++p) {
                            float px = (pred_data[5 + p * 2] * anchor[a_id * 2] + w_id * stride);
                            float py = (pred_data[5 + p * 2 + 1] * anchor[a_id * 2 + 1] + h_id * stride);
                            // px = std::max(std::min(px, (float)(INPUT_W)), 0.f);
                            // py = std::max(std::min(py, (float)(INPUT_H)), 0.f);
                            now.pts[p] = cv::Point2f(px, py);
                            // std::cout << px << " " << py  << " ";
                        }
                        // std::cout << std::endl;

                        float x0 = (x - w * 0.5);
                        float y0 = (y - h * 0.5);
                        float x1 = (x + w * 0.5);
                        float y1 = (y + h * 0.5);

                        // x0 = std::max(std::min(x0, (float)(INPUT_W)), 0.f);
                        // y0 = std::max(std::min(y0, (float)(INPUT_H)), 0.f);
                        // x1 = std::max(std::min(x1, (float)(INPUT_W)), 0.f);
                        // y1 = std::max(std::min(y1, (float)(INPUT_H)), 0.f);

                        now.rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);
                        now.conf = final_conf;
                        now.color = col_id;
                        now.type = cls_id;
                        now.size = ts_id;
                        objects.push_back(now);
                    }
                }
            }
        }
    }
    return;
}

YOLOv5FlatDecoder::YOLOv5FlatDecoder(toml::value &config, const rclcpp::Logger &_logger)
    : NetDecoderBase(config, _logger) {
    class_color_map = toml::get<std::vector<int>>(config.at("class_color_map"));
    class_type_map = toml::get<std::vector<int>>(config.at("class_type_map"));
    assert((int)class_color_map.size() == NUM_CLASSES && "class_color_map size mismatch");
    assert((int)class_type_map.size() == NUM_CLASSES && "class_type_map size mismatch");
}

void YOLOv5FlatDecoder::set_layer_info(int layer_index, const std::vector<size_t> &dims) {
    if ((int)layers.size() != layer_index) {
        throw std::runtime_error("YOLOv5FlatDecoder layer info must be set in order");
    }
    if (dims.size() != 3) {
        throw std::runtime_error("YOLOv5FlatDecoder expected 3-dim output [batch, num_preds, "
                                 "num_outputs], got " + dims_to_string(dims));
    }
    YOLOv5FlatLayerInfo layer = {
        layer_index,
        static_cast<int>(dims[1]),
        static_cast<int>(dims[2]),
    };
    RCLCPP_INFO(logger, "flat layer %d: num_preds=%d, num_outputs=%d", layer_index, layer.num_preds,
                layer.num_outputs);
    if (!this->check_num_outputs(layer.num_outputs)) {
        throw std::runtime_error("YOLOv5FlatDecoder expected num_outputs=" +
                                 std::to_string(5 + NUM_CLASSES) + ", got " +
                                 std::to_string(layer.num_outputs));
    }
    layers.push_back(layer);
}

bool YOLOv5FlatDecoder::check_num_outputs(int num_outputs) { return num_outputs == 5 + NUM_CLASSES; }

void YOLOv5FlatDecoder::decode(int layer_index, const float *prob, std::vector<Armor> &objects) {
    assert((int)layers.size() > layer_index && "layer_index out of range");
    auto [_, num_preds, no] = layers[layer_index];
    ++decode_calls;

    float max_raw_obj_conf = -std::numeric_limits<float>::infinity();
    float max_sigmoid_obj_conf = -std::numeric_limits<float>::infinity();
    float max_raw_cls_conf = -std::numeric_limits<float>::infinity();
    float max_sigmoid_cls_conf = -std::numeric_limits<float>::infinity();
    int best_pred_idx = -1;
    int best_cls_id = -1;
    int invalid_candidate_count = 0;
    int low_conf_candidate_count = 0;
    int nonfinite_pred_count = 0;
    float best_raw_vals[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    float best_sigmoid_vals[2] = {0.f, 0.f};
    bool captured_invalid_candidate = false;
    float invalid_vals[5] = {0.f, 0.f, 0.f, 0.f, 0.f};

    for (int pred_idx = 0; pred_idx < num_preds; ++pred_idx) {
        const float *pred = prob + pred_idx * no;
        bool pred_finite = true;
        for (int idx = 0; idx < no; ++idx) {
            if (!std::isfinite(pred[idx])) {
                pred_finite = false;
                break;
            }
        }
        if (!pred_finite) {
            ++nonfinite_pred_count;
            continue;
        }

        const float raw_obj_conf = pred[4];
        const float sigmoid_obj_conf = sigmoid(raw_obj_conf);
        if (std::isfinite(raw_obj_conf)) {
            max_raw_obj_conf = std::max(max_raw_obj_conf, raw_obj_conf);
        }
        if (std::isfinite(sigmoid_obj_conf)) {
            max_sigmoid_obj_conf = std::max(max_sigmoid_obj_conf, sigmoid_obj_conf);
        }

        int cls_id = 0;
        float raw_cls_conf = pred[5];
        float sigmoid_cls_conf = sigmoid(raw_cls_conf);
        for (int idx = 1; idx < NUM_CLASSES; ++idx) {
            if (pred[5 + idx] > raw_cls_conf) {
                raw_cls_conf = pred[5 + idx];
                sigmoid_cls_conf = sigmoid(raw_cls_conf);
                cls_id = idx;
            }
        }
        if (std::isfinite(raw_cls_conf)) {
            max_raw_cls_conf = std::max(max_raw_cls_conf, raw_cls_conf);
        }
        if (std::isfinite(sigmoid_cls_conf)) {
            max_sigmoid_cls_conf = std::max(max_sigmoid_cls_conf, sigmoid_cls_conf);
        }

        if (raw_obj_conf > max_raw_obj_conf - 1e-6f) {
            best_pred_idx = pred_idx;
            best_cls_id = cls_id;
            best_raw_vals[0] = pred[0];
            best_raw_vals[1] = pred[1];
            best_raw_vals[2] = pred[2];
            best_raw_vals[3] = pred[3];
            best_raw_vals[4] = raw_obj_conf;
            best_sigmoid_vals[0] = sigmoid_obj_conf;
            best_sigmoid_vals[1] = sigmoid_cls_conf;
        }

        const float obj_conf = raw_obj_conf;
        if (obj_conf <= BBOX_CONF_THRESH)
        {
            ++low_conf_candidate_count;
            continue;
        }
        const float cls_conf = raw_cls_conf;

        const double final_conf = obj_conf * cls_conf;
        if (final_conf <= BBOX_CONF_THRESH)
            continue;
        if (cls_id < 0 || cls_id >= NUM_CLASSES)
            continue;

        const int color = class_color_map[cls_id];
        const int type = class_type_map[cls_id];
        if (color < 0 || type < 0)
            continue;

        const float cx = pred[0];
        const float cy = pred[1];
        const float w = pred[2];
        const float h = pred[3];
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) ||
            !std::isfinite(final_conf) || w <= 0.f || h <= 0.f) {
            ++invalid_candidate_count;
            if (!captured_invalid_candidate) {
                invalid_vals[0] = cx;
                invalid_vals[1] = cy;
                invalid_vals[2] = w;
                invalid_vals[3] = h;
                invalid_vals[4] = static_cast<float>(final_conf);
                captured_invalid_candidate = true;
            }
            continue;
        }
        const float x0 = cx - w * 0.5f;
        const float y0 = cy - h * 0.5f;
        const float x1 = cx + w * 0.5f;
        const float y1 = cy + h * 0.5f;
        if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1) ||
            x1 <= x0 || y1 <= y0) {
            ++invalid_candidate_count;
            if (!captured_invalid_candidate) {
                invalid_vals[0] = cx;
                invalid_vals[1] = cy;
                invalid_vals[2] = w;
                invalid_vals[3] = h;
                invalid_vals[4] = static_cast<float>(final_conf);
                captured_invalid_candidate = true;
            }
            continue;
        }

        Armor now;
        now.rect = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        now.conf = final_conf;
        now.color = color;
        now.type = type;
        now.size = 0;
        now.pts[0] = cv::Point2f(x0, y0);
        now.pts[1] = cv::Point2f(x1, y0);
        now.pts[2] = cv::Point2f(x1, y1);
        now.pts[3] = cv::Point2f(x0, y1);
        now.pts[4] = cv::Point2f(cx, cy);
        objects.push_back(now);
    }

    if (decode_calls % debug_log_period == 0 &&
        (invalid_candidate_count > 0 || nonfinite_pred_count > 0 ||
         (objects.empty() && max_raw_obj_conf > BBOX_CONF_THRESH))) {
        std::string invalid_sample_suffix;
        if (captured_invalid_candidate) {
            invalid_sample_suffix = " invalid_sample=[" + std::to_string(invalid_vals[0]) + " " +
                                    std::to_string(invalid_vals[1]) + " " +
                                    std::to_string(invalid_vals[2]) + " " +
                                    std::to_string(invalid_vals[3]) + " " +
                                    std::to_string(invalid_vals[4]) + "]";
        }
        RCLCPP_WARN(
            logger,
            "V5_FLAT decode diag: layer=%d preds=%d kept=%zu nonfinite=%d invalid=%d low_conf=%d max_raw_obj=%.4f "
            "max_sigmoid_obj=%.4f max_raw_cls=%.4f max_sigmoid_cls=%.4f best_pred=%d best_cls=%d "
            "best_raw=[%.4f %.4f %.4f %.4f %.4f] best_sigmoid=[%.4f %.4f]%s",
            layer_index, num_preds, objects.size(), nonfinite_pred_count, invalid_candidate_count,
            low_conf_candidate_count,
            max_raw_obj_conf, max_sigmoid_obj_conf, max_raw_cls_conf, max_sigmoid_cls_conf,
            best_pred_idx, best_cls_id, best_raw_vals[0], best_raw_vals[1], best_raw_vals[2],
            best_raw_vals[3], best_raw_vals[4], best_sigmoid_vals[0], best_sigmoid_vals[1],
            invalid_sample_suffix.c_str());
    }
}

DETRDecoder::DETRDecoder(toml::value &config, const rclcpp::Logger &_logger)
    : NetDecoderBase(config, _logger) {
    try {
        min_class_score = static_cast<float>(config.at("DETR_MIN_CLASS_SCORE").as_floating());
    } catch (...) {
        min_class_score = 0.0f;
    }
    try {
        min_color_score = static_cast<float>(config.at("DETR_MIN_COLOR_SCORE").as_floating());
    } catch (...) {
        min_color_score = 0.0f;
    }
    try {
        class_margin = static_cast<float>(config.at("DETR_CLASS_MARGIN").as_floating());
    } catch (...) {
        class_margin = 0.0f;
    }
    try {
        color_margin = static_cast<float>(config.at("DETR_COLOR_MARGIN").as_floating());
    } catch (...) {
        color_margin = 0.0f;
    }
}

void DETRDecoder::set_layer_info(int layer_index, const std::vector<size_t> &dims) {
    assert((int)layers.size() == layer_index && "set_layer_info should be called in order");
    assert(dims.size() == 3 && "DETR model output should be 3-dim [batch, queries, outputs]");

    DETRLayerInfo layer = {
        layer_index,
        static_cast<int>(dims[1]),  // num_queries
        static_cast<int>(dims[2]),  // num_outputs per query
    };
    RCLCPP_INFO(logger, "[DETR] layer %d: num_queries=%d, num_outputs=%d",
                layer_index, layer.num_queries, layer.num_outputs);
    assert(check_num_outputs(layer.num_outputs) && "DETR num_output check failed");
    layers.push_back(layer);
}

bool DETRDecoder::check_num_outputs(int num_outputs) {
    return num_outputs == 4 + NUM_CLASSES + NUM_COLORS;
}

void DETRDecoder::decode(int layer_index, const float *prob, std::vector<Armor> &objects) {
    assert((int)layers.size() > layer_index && "layer_index out of range");
    const auto &layer = layers[layer_index];
    const int no = layer.num_outputs;
    const int nq = layer.num_queries;

    for (int q = 0; q < nq; ++q) {
        const float *row = prob + q * no;
        // bbox: cx, cy, w, h (normalized 0-1 after sigmoid)
        float cx = row[0] * INPUT_W;
        float cy = row[1] * INPUT_H;
        float bw  = row[2] * INPUT_W;
        float bh  = row[3] * INPUT_H;

        // class and color scores (already sigmoid)
        const float *cls_scores = row + 4;
        const float *col_scores = row + 4 + NUM_CLASSES;

        int cls_id = 0;
        int cls_second_id = 0;
        float cls_best = cls_scores[0];
        float cls_second = -1.0f;
        for (int idx = 1; idx < NUM_CLASSES; ++idx) {
            if (cls_scores[idx] > cls_best) {
                cls_second = cls_best;
                cls_second_id = cls_id;
                cls_best = cls_scores[idx];
                cls_id = idx;
            } else if (cls_scores[idx] > cls_second) {
                cls_second = cls_scores[idx];
                cls_second_id = idx;
            }
        }

        int col_id = 0;
        int col_second_id = 0;
        float col_best = col_scores[0];
        float col_second = -1.0f;
        for (int idx = 1; idx < NUM_COLORS; ++idx) {
            if (col_scores[idx] > col_best) {
                col_second = col_best;
                col_second_id = col_id;
                col_best = col_scores[idx];
                col_id = idx;
            } else if (col_scores[idx] > col_second) {
                col_second = col_scores[idx];
                col_second_id = idx;
            }
        }

        if (cls_best < min_class_score)
            continue;
        if (col_best < min_color_score)
            continue;
        if (cls_second_id != cls_id && cls_best - cls_second < class_margin)
            continue;
        if (col_second_id != col_id && col_best - col_second < color_margin)
            continue;

        // Use geometric mean to avoid over-penalizing class/color joint confidence.
        float final_conf = std::sqrt(std::max(0.0f, cls_best * col_best));
        if (final_conf <= BBOX_CONF_THRESH)
            continue;

        // Matcher only consumes type [0..5]. Skip other classes (0/Bs/Bb-like) here.
        if (cls_id > 5)
            continue;

        // Matcher only consumes BLUE/RED colors. Skip N/P-like classes here.
        if (col_id > 1)
            continue;

        Armor now;
        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;
        x0 = std::max(0.0f, std::min(x0, static_cast<float>(INPUT_W - 1)));
        y0 = std::max(0.0f, std::min(y0, static_cast<float>(INPUT_H - 1)));
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(INPUT_W - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(INPUT_H - 1)));
        if (x1 <= x0 || y1 <= y0)
            continue;
        now.rect  = cv::Rect(x0, y0, x1 - x0, y1 - y0);
        now.conf  = final_conf;
        now.color = col_id;
        now.type  = cls_id;
        now.size  = 0;
        // derive 4 corner pts + center (pts[4])
        now.pts[0] = cv::Point2f(x0, y0);
        now.pts[1] = cv::Point2f(x1, y0);
        now.pts[2] = cv::Point2f(x1, y1);
        now.pts[3] = cv::Point2f(x0, y1);
        now.pts[4] = cv::Point2f(cx, cy);
        objects.push_back(now);
    }
}

YOLOv8Decoder::YOLOv8Decoder(toml::value &config, const rclcpp::Logger& _logger) : NetDecoderBase(config, _logger) {
    if (config.contains("standard_detect")) {
        standard_detect = config.at("standard_detect").as_boolean();
    }
    if (standard_detect) {
        NUM_KPTS = 4;
        NUM_TSIZES = 0;
        class_color_map = toml::get<std::vector<int>>(config.at("class_color_map"));
        class_type_map = toml::get<std::vector<int>>(config.at("class_type_map"));
        if ((int)class_color_map.size() != NUM_CLASSES || (int)class_type_map.size() != NUM_CLASSES) {
            throw std::runtime_error("YOLOv8 standard class maps must match NUM_CLASSES");
        }
    } else {
        NUM_KPTS = config.at("NUM_KPTS").as_integer();
        NUM_TSIZES = config.at("NUM_TSIZES").as_integer();
    }
}

void YOLOv8Decoder::set_layer_info(int layer_index, const std::vector<size_t> &dims) {
    assert((int)layers.size() == layer_index && "set_layer_info should be called in order");
    assert(dims.size() == 3 && "model should be 3-dim");
    bool channels_first = false;
    int num_preds = 0;
    int num_outputs = 0;
    if (dims[1] == 8400) {
        num_preds = static_cast<int>(dims[1]);
        num_outputs = static_cast<int>(dims[2]);
    } else if (dims[2] == 8400) {
        channels_first = true;
        num_preds = static_cast<int>(dims[2]);
        num_outputs = static_cast<int>(dims[1]);
    } else {
        throw std::runtime_error("YOLOv8 output should be [1, 8400, no] or [1, no, 8400], got " + dims_to_string(dims));
    }

    // YOLOv8LayerInfo layer{
    //     .index = layer_index,
    //     .num_outputs = static_cast<int>(dims[2]),
    // };
    YOLOv8LayerInfo layer = {layer_index, num_outputs, num_preds, 0, channels_first};

    RCLCPP_INFO(logger, "layer %d: num_preds=%d num_outputs=%d channels_first=%s",
                layer_index, layer.num_preds, layer.num_outputs,
                layer.channels_first ? "true" : "false");
    assert(this->check_num_outputs(layer.num_outputs) && "num_output check failed");

    layers.push_back(layer);
    return;
}

bool YOLOv8Decoder::check_num_outputs(int num_outputs) {
    if (standard_detect) {
        return num_outputs == 4 + NUM_CLASSES;
    }
    return num_outputs == NUM_KPTS + NUM_CLASSES + NUM_COLORS + NUM_TSIZES;
}

void YOLOv8Decoder::decode(int layer_index, const float *prob, std::vector<Armor> &objects) {
    assert((int)layers.size() > layer_index && "layer_index out of range");
    const auto &layer = layers[layer_index];
    int no = layer.num_outputs;
    int num_preds = layer.num_preds;

    auto at = [&](int idx, int channel) -> float {
        if (layer.channels_first) {
            return prob[channel * num_preds + idx];
        }
        return prob[idx * no + channel];
    };

    if (standard_detect) {
        for (int idx = 0; idx < num_preds; ++idx) {
            int cls_id = 0;
            float cls_best = at(idx, 4);
            for (int c = 1; c < NUM_CLASSES; ++c) {
                float score = at(idx, 4 + c);
                if (score > cls_best) {
                    cls_best = score;
                    cls_id = c;
                }
            }
            if (cls_best <= BBOX_CONF_THRESH) {
                continue;
            }

            const float cx = at(idx, 0);
            const float cy = at(idx, 1);
            const float bw = at(idx, 2);
            const float bh = at(idx, 3);
            float x0 = std::max(0.0f, std::min(cx - bw * 0.5f, static_cast<float>(INPUT_W - 1)));
            float y0 = std::max(0.0f, std::min(cy - bh * 0.5f, static_cast<float>(INPUT_H - 1)));
            float x1 = std::max(0.0f, std::min(cx + bw * 0.5f, static_cast<float>(INPUT_W - 1)));
            float y1 = std::max(0.0f, std::min(cy + bh * 0.5f, static_cast<float>(INPUT_H - 1)));
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }

            Armor now;
            now.pts[0] = cv::Point2f(x0, y0);
            now.pts[1] = cv::Point2f(x1, y0);
            now.pts[2] = cv::Point2f(x1, y1);
            now.pts[3] = cv::Point2f(x0, y1);
            now.pts[4] = cv::Point2f(cx, cy);
            now.rect = cv::Rect(cv::Point2f(x0, y0), cv::Point2f(x1, y1));
            now.conf = cls_best;
            now.color = class_color_map[cls_id];
            now.type = class_type_map[cls_id];
            now.size = 0;
            objects.push_back(now);
        }
        return;
    }

    std::vector<float> pred_data_v;
    pred_data_v.resize(no);
    float* pred_data = pred_data_v.data();
    // [kpts(8), hot(classes)(8), hot(tsizes)(2), hot(colors)(4)]
    for (int idx = 0; idx < num_preds; ++idx) {
        float rough_conf = *std::max_element(&prob[idx * no + NUM_KPTS], &prob[idx * no + no]);

        if (rough_conf > BBOX_CONF_THRESH) {
            std::memcpy(pred_data, &prob[idx * no], no * sizeof(float));
            int cls_id = std::distance(pred_data + NUM_KPTS, std::max_element(pred_data + NUM_KPTS, pred_data + NUM_KPTS + NUM_CLASSES));
            int ts_id = std::distance(pred_data + NUM_KPTS + NUM_CLASSES, std::max_element(pred_data + NUM_KPTS + NUM_CLASSES, pred_data + NUM_KPTS + NUM_CLASSES + NUM_TSIZES));
            int col_id = std::distance(pred_data + NUM_KPTS + NUM_CLASSES + NUM_TSIZES,
                                       std::max_element(pred_data + NUM_KPTS + NUM_CLASSES + NUM_TSIZES, pred_data + NUM_KPTS + NUM_CLASSES + NUM_TSIZES + NUM_COLORS));

            double final_conf = std::min({pred_data[NUM_KPTS + NUM_CLASSES + NUM_TSIZES + col_id], pred_data[NUM_KPTS + cls_id]});
            if (final_conf > BBOX_CONF_THRESH) {
                // std::cout << final_conf << " " << col_id << " "
                //           << cls_id << std::endl;
                Armor now;

                for (int p = 0; p < (NUM_KPTS / 2); ++p) {
                    // float px = std::max(std::min(pred_data[p * 2], (float)(INPUT_W)), 0.f);
                    // float py = std::max(std::min(pred_data[p * 2 + 1], (float)(INPUT_H)), 0.f);
                    float px = pred_data[p * 2];
                    float py = pred_data[p * 2 + 1];
                    now.pts[p] = cv::Point2f(px, py);
                }

                now.rect = cv::Rect(now.pts[0], now.pts[2]);
                now.conf = final_conf;
                now.color = col_id;
                now.type = cls_id;
                now.size = ts_id;
                objects.push_back(now);
            }
        }
    }
}
