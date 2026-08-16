#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/centroid.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/u_int16.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <locate/locator.h>
#include <robot/robot.h>
#include <track/tracker.h>
#include <radar_interface/msg/armor.hpp>
#include <radar_interface/msg/detected_target_array.hpp>
#include <radar_interface/msg/match_result.hpp>
#include <radar_interface/msg/target_array.hpp>
#include <radar_interface/team_color.hpp>

namespace {

constexpr int kClassNum = 12;
constexpr int kRobotsPerTeam = 6;
constexpr double kFieldWidthM = 28.0;
constexpr double kFieldHeightM = 15.0;

struct DnnDetection {
    cv::Rect2f box;
    int label = -1;
    int color = -1;
    int type = -1;
    float confidence = 0.0f;
};

std::vector<DnnDetection> parse_yolo_output(const cv::Mat& output, const cv::Size& model_size, const cv::Size& image_size,
    int class_num, const std::vector<int>& class_filter, float conf_threshold, float nms_threshold)
{
    cv::Mat out = output;
    if (out.dims == 3) {
        const int rows = out.size[1];
        const int cols = out.size[2];
        out = cv::Mat(rows, cols, CV_32F, out.ptr<float>());
        if (rows < cols)
            cv::transpose(out, out);
    } else if (out.dims == 2) {
        out = out.reshape(1, out.size[0]);
    } else {
        return {};
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    for (int i = 0; i < out.rows; ++i) {
        const float* row = out.ptr<float>(i);
        const int score_offset = 4;
        int best_label = -1;
        float best_score = 0.0f;
        const int available_classes = std::min(class_num, out.cols - score_offset);
        for (int c = 0; c < available_classes; ++c) {
            if (row[score_offset + c] > best_score) {
                best_score = row[score_offset + c];
                best_label = c;
            }
        }
        if (best_score < conf_threshold)
            continue;
        if (!class_filter.empty()
            && std::find(class_filter.begin(), class_filter.end(), best_label) == class_filter.end())
            continue;

        const float sx = static_cast<float>(image_size.width) / static_cast<float>(model_size.width);
        const float sy = static_cast<float>(image_size.height) / static_cast<float>(model_size.height);
        const float cx = row[0] * sx;
        const float cy = row[1] * sy;
        const float w = row[2] * sx;
        const float h = row[3] * sy;
        cv::Rect2f box(cx - w * 0.5f, cy - h * 0.5f, w, h);
        box &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(image_size.width), static_cast<float>(image_size.height));
        if (box.empty())
            continue;
        boxes.emplace_back(cv::Rect(cvRound(box.x), cvRound(box.y), cvRound(box.width), cvRound(box.height)));
        scores.push_back(best_score);
        labels.push_back(best_label);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, keep);
    std::vector<DnnDetection> detections;
    detections.reserve(keep.size());
    for (int idx : keep)
        detections.push_back({ boxes[idx], labels[idx], -1, -1, scores[idx] });
    return detections;
}

std::vector<DnnDetection> parse_armor_output(const cv::Mat& output, const cv::Size& model_size, const cv::Size& image_size,
    const std::vector<int>& class_color_map, const std::vector<int>& class_type_map,
    float conf_threshold, float nms_threshold)
{
    cv::Mat out = output;
    if (out.dims != 3)
        return {};

    const int dim1 = out.size[1];
    const int dim2 = out.size[2];
    const int class_num = static_cast<int>(std::min(class_color_map.size(), class_type_map.size()));
    if (class_num <= 0)
        return {};

    const int channels_without_obj = 4 + class_num;
    const int channels_with_obj = 5 + class_num;
    const bool channels_first = dim1 == channels_without_obj || dim1 == channels_with_obj;
    const int channels = channels_first ? dim1 : dim2;
    const int preds = channels_first ? dim2 : dim1;
    if (channels != channels_without_obj && channels != channels_with_obj)
        return {};
    const bool has_objectness = channels == channels_with_obj;
    const int class_offset = has_objectness ? 5 : 4;

    auto at = [&](int idx, int channel) -> float {
        const float* ptr = out.ptr<float>();
        if (channels_first)
            return ptr[channel * preds + idx];
        return ptr[idx * channels + channel];
    };

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    for (int i = 0; i < preds; ++i) {
        int best_label = 0;
        float best_class_score = at(i, class_offset);
        for (int c = 1; c < class_num; ++c) {
            const float class_score = at(i, class_offset + c);
            if (class_score > best_class_score) {
                best_class_score = class_score;
                best_label = c;
            }
        }
        const float objectness = has_objectness ? at(i, 4) : 1.0f;
        const float best_score = objectness * best_class_score;
        if (best_score < conf_threshold)
            continue;

        const float sx = static_cast<float>(image_size.width) / static_cast<float>(model_size.width);
        const float sy = static_cast<float>(image_size.height) / static_cast<float>(model_size.height);
        const float cx = at(i, 0) * sx;
        const float cy = at(i, 1) * sy;
        const float w = at(i, 2) * sx;
        const float h = at(i, 3) * sy;
        cv::Rect2f box(cx - w * 0.5f, cy - h * 0.5f, w, h);
        box &= cv::Rect2f(0.0f, 0.0f, static_cast<float>(image_size.width), static_cast<float>(image_size.height));
        if (box.empty())
            continue;
        boxes.emplace_back(cv::Rect(cvRound(box.x), cvRound(box.y), cvRound(box.width), cvRound(box.height)));
        scores.push_back(best_score);
        labels.push_back(best_label);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, keep);
    std::vector<DnnDetection> detections;
    detections.reserve(keep.size());
    for (int idx : keep) {
        const int label = labels[idx];
        detections.push_back({ boxes[idx], label, class_color_map[label], class_type_map[label], scores[idx] });
    }
    return detections;
}

class CvDnnRobotDetector {
public:
    CvDnnRobotDetector(const std::string& car_model, const std::string& armor_model, cv::Size input_size,
        int class_num, const std::vector<int>& armor_class_color_map, const std::vector<int>& armor_class_type_map,
        int car_class_num, const std::vector<int>& car_class_filter,
        float car_conf, float armor_conf, float nms)
        : input_size_(input_size)
        , class_num_(class_num)
        , armor_class_color_map_(armor_class_color_map)
        , armor_class_type_map_(armor_class_type_map)
        , car_class_num_(car_class_num)
        , car_class_filter_(car_class_filter)
        , car_conf_(car_conf)
        , armor_conf_(armor_conf)
        , nms_(nms)
    {
        car_net_ = cv::dnn::readNet(car_model);
        armor_net_ = cv::dnn::readNet(armor_model);
        car_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        armor_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        car_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        armor_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::vector<radar::Robot> detect(const cv::Mat& image)
    {
        std::vector<radar::Robot> robots;
        if (image.empty())
            return robots;
        auto cars = infer(car_net_, image, car_class_num_, car_class_filter_, car_conf_);
        robots.reserve(cars.size());
        for (const auto& car : cars) {
            cv::Rect car_rect = car.box;
            car_rect &= cv::Rect(0, 0, image.cols, image.rows);
            if (car_rect.empty())
                continue;
            cv::Mat crop = image(car_rect).clone();
            auto armors = infer_armor(armor_net_, crop);
            std::vector<radar::Detection> armor_detections;
            armor_detections.reserve(armors.size());
            for (const auto& armor : armors) {
                const int label = color_type_to_label(armor.color, armor.type);
                if (label < 0)
                    continue;
                armor_detections.emplace_back(armor.box.x, armor.box.y, armor.box.width, armor.box.height, label, armor.confidence);
            }
            radar::Detection car_detection(car_rect.x, car_rect.y, car_rect.width, car_rect.height, 0, car.confidence);
            robots.emplace_back(car_detection, armor_detections);
        }
        return robots;
    }

private:
    std::vector<DnnDetection> infer(cv::dnn::Net& net, const cv::Mat& image, int class_num,
        const std::vector<int>& class_filter, float conf)
    {
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();
        return parse_yolo_output(output, input_size_, image.size(), class_num, class_filter, conf, nms_);
    }

    std::vector<DnnDetection> infer_armor(cv::dnn::Net& net, const cv::Mat& image)
    {
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();
        return parse_armor_output(output, input_size_, image.size(), armor_class_color_map_, armor_class_type_map_,
            armor_conf_, nms_);
    }

    int color_type_to_label(int color, int type) const
    {
        if (color == 0) {
            switch (type) {
            case 0: return radar::Label::BlueSentry;
            case 1: return radar::Label::BlueHero;
            case 2: return radar::Label::BlueEngineer;
            case 3: return radar::Label::BlueInfantryThree;
            case 4: return radar::Label::BlueInfantryFour;
            case 5: return radar::Label::BlueInfantryFive;
            default: return -1;
            }
        }
        if (color == 1) {
            switch (type) {
            case 0: return radar::Label::RedSentry;
            case 1: return radar::Label::RedHero;
            case 2: return radar::Label::RedEngineer;
            case 3: return radar::Label::RedInfantryThree;
            case 4: return radar::Label::RedInfantryFour;
            case 5: return radar::Label::RedInfantryFive;
            default: return -1;
            }
        }
        return -1;
    }

    cv::dnn::Net car_net_;
    cv::dnn::Net armor_net_;
    cv::Size input_size_;
    int class_num_;
    std::vector<int> armor_class_color_map_;
    std::vector<int> armor_class_type_map_;
    int car_class_num_ = 1;
    std::vector<int> car_class_filter_;
    float car_conf_;
    float armor_conf_;
    float nms_;
};

cv::Matx44f to_scaled_mat(const geometry_msgs::msg::TransformStamped& tf_msg, float translation_scale)
{
    const auto iso = tf2::transformToEigen(tf_msg);
    cv::Matx44f mat = cv::Matx44f::eye();
    const auto& m = iso.matrix();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            mat(r, c) = static_cast<float>(m(r, c));
        mat(r, 3) = static_cast<float>(m(r, 3)) * translation_scale;
    }
    return mat;
}

bool transform_changed(const Eigen::Matrix4d& last, const Eigen::Matrix4d& current)
{
    const double translation_delta = (last.block<3, 1>(0, 3) - current.block<3, 1>(0, 3)).norm();
    const double rotation_delta = (last.block<3, 3>(0, 0) - current.block<3, 3>(0, 0)).norm();
    return translation_delta > 1e-3 || rotation_delta > 1e-4;
}

radar_interface::team_color::ENUM enemy_color_for(radar_interface::team_color::ENUM ally_color)
{
    if (ally_color == radar_interface::team_color::C_RED)
        return radar_interface::team_color::C_BLUE;
    if (ally_color == radar_interface::team_color::C_BLUE)
        return radar_interface::team_color::C_RED;
    return radar_interface::team_color::UNKNOWN;
}

const char* color_name(radar_interface::team_color::ENUM color)
{
    switch (color) {
    case radar_interface::team_color::C_RED:
        return "red";
    case radar_interface::team_color::C_BLUE:
        return "blue";
    default:
        return "unknown";
    }
}

cv::Rect clamp_rect_to_image(const cv::Rect& rect, const cv::Mat& image)
{
    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    return rect & image_rect;
}

bool has_expected_light(const cv::Mat& image, const cv::Mat& roi_mask,
    radar_interface::team_color::ENUM expected_color,
    int min_light_pixels, double min_light_ratio,
    int min_saturation, int min_value)
{
    if (image.empty() || roi_mask.empty() || roi_mask.size() != image.size() ||
        roi_mask.type() != CV_8UC1 || expected_color == radar_interface::team_color::UNKNOWN) {
        return false;
    }

    cv::Rect roi = cv::boundingRect(roi_mask);
    roi = clamp_rect_to_image(roi, image);
    if (roi.empty())
        return false;

    const cv::Mat mask_roi = roi_mask(roi);
    const int roi_pixels = cv::countNonZero(mask_roi);
    if (roi_pixels <= 0)
        return false;

    cv::Mat hsv;
    cv::cvtColor(image(roi), hsv, cv::COLOR_BGR2HSV);

    cv::Mat color_mask;
    if (expected_color == radar_interface::team_color::C_RED) {
        cv::Mat lower_red;
        cv::Mat upper_red;
        cv::inRange(hsv, cv::Scalar(0, min_saturation, min_value), cv::Scalar(10, 255, 255), lower_red);
        cv::inRange(hsv, cv::Scalar(170, min_saturation, min_value), cv::Scalar(180, 255, 255), upper_red);
        color_mask = lower_red | upper_red;
    } else {
        cv::inRange(hsv, cv::Scalar(95, min_saturation, min_value), cv::Scalar(135, 255, 255), color_mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, kernel);
    cv::bitwise_and(color_mask, mask_roi, color_mask);

    const int light_pixels = cv::countNonZero(color_mask);
    const double light_ratio = static_cast<double>(light_pixels) / static_cast<double>(roi_pixels);
    return light_pixels >= min_light_pixels && light_ratio >= min_light_ratio;
}

std::chrono::high_resolution_clock::time_point to_chrono_time(const rclcpp::Time& stamp)
{
    return std::chrono::high_resolution_clock::time_point(std::chrono::nanoseconds(stamp.nanoseconds()));
}

std::optional<std::pair<int, int>> label_to_color_type(int label)
{
    using Armor = radar_interface::msg::Armor;
    auto ct = [](int color, int type) {
        return std::optional<std::pair<int, int>>(std::make_pair(color, type));
    };
    switch (label) {
    case radar::Label::BlueHero:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_HERO);
    case radar::Label::BlueEngineer:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_ENGINEER);
    case radar::Label::BlueInfantryThree:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_INF_3);
    case radar::Label::BlueInfantryFour:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_INF_4);
    case radar::Label::BlueInfantryFive:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_INF_5);
    case radar::Label::BlueSentry:
        return ct(Armor::COLOR_BLUE, Armor::TYPE_SENTRY);
    case radar::Label::RedHero:
        return ct(Armor::COLOR_RED, Armor::TYPE_HERO);
    case radar::Label::RedEngineer:
        return ct(Armor::COLOR_RED, Armor::TYPE_ENGINEER);
    case radar::Label::RedInfantryThree:
        return ct(Armor::COLOR_RED, Armor::TYPE_INF_3);
    case radar::Label::RedInfantryFour:
        return ct(Armor::COLOR_RED, Armor::TYPE_INF_4);
    case radar::Label::RedInfantryFive:
        return ct(Armor::COLOR_RED, Armor::TYPE_INF_5);
    case radar::Label::RedSentry:
        return ct(Armor::COLOR_RED, Armor::TYPE_SENTRY);
    default:
        return std::nullopt;
    }
}

std::string label_name(int label)
{
    switch (label) {
    case radar::Label::BlueHero:
        return "B-HERO";
    case radar::Label::BlueEngineer:
        return "B-ENG";
    case radar::Label::BlueInfantryThree:
        return "B-3";
    case radar::Label::BlueInfantryFour:
        return "B-4";
    case radar::Label::BlueInfantryFive:
        return "B-5";
    case radar::Label::BlueSentry:
        return "B-SENTRY";
    case radar::Label::RedHero:
        return "R-HERO";
    case radar::Label::RedEngineer:
        return "R-ENG";
    case radar::Label::RedInfantryThree:
        return "R-3";
    case radar::Label::RedInfantryFour:
        return "R-4";
    case radar::Label::RedInfantryFive:
        return "R-5";
    case radar::Label::RedSentry:
        return "R-SENTRY";
    default:
        return "UNKNOWN";
    }
}

int type_to_slot(int type)
{
    if (type < radar_interface::msg::Armor::TYPE_SENTRY || type > radar_interface::msg::Armor::TYPE_INF_5)
        return -1;
    return type;
}

} // namespace

class RmRadarPipelineNode : public rclcpp::Node {
public:
    RmRadarPipelineNode()
        : Node("rm_radar_pipeline")
        , tf_buffer_(this->get_clock())
        , tf_listener_(tf_buffer_)
    {
        image_topic_ = declare_parameter("image_topic", std::string("hik_6mm/image"));
        camera_info_topic_ = declare_parameter("camera_info_topic", std::string("hik_6mm/camera_info"));
        pointcloud_topic_ = declare_parameter("pointcloud_topic", std::string("lidar_mid70/pc_raw"));
        camera_frame_ = declare_parameter("camera_frame", std::string("hik_6mm_frame"));
        lidar_frame_ = declare_parameter("lidar_frame", std::string("lidar_mid70_frame"));
        world_frame_ = declare_parameter("world_frame", std::string("world"));
        car_model_path_ = declare_parameter("car_model_path", default_model_path("car.onnx"));
        car_class_num_ = declare_parameter("car_class_num", 1);
        const auto car_class_filter_param =
            declare_parameter<std::vector<int64_t>>("car_class_filter", std::vector<int64_t> {});
        car_class_filter_.assign(car_class_filter_param.begin(), car_class_filter_param.end());
        armor_model_path_ = declare_parameter("armor_model_path", default_nn_detector_model_path("armor_v8.onnx"));
        const auto armor_class_color_map_param = declare_parameter<std::vector<int64_t>>(
            "armor_class_color_map", {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1});
        const auto armor_class_type_map_param = declare_parameter<std::vector<int64_t>>(
            "armor_class_type_map", {1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0});
        armor_class_color_map_.assign(armor_class_color_map_param.begin(), armor_class_color_map_param.end());
        armor_class_type_map_.assign(armor_class_type_map_param.begin(), armor_class_type_map_param.end());
        min_publish_confidence_ = declare_parameter("min_publish_confidence", 0.0);
        car_confidence_threshold_ = declare_parameter("car_confidence_threshold", 0.25);
        armor_confidence_threshold_ = declare_parameter("armor_confidence_threshold", 0.50);
        nms_threshold_ = declare_parameter("nms_threshold", 0.65);
        publish_tentative_tracks_ = declare_parameter("publish_tentative_tracks", false);
        camera_fallback_enabled_ = declare_parameter("camera_fallback_enabled", false);
        camera_fallback_hold_ms_ = declare_parameter("camera_fallback_hold_ms", 500);
        camera_fallback_min_confidence_ = declare_parameter("camera_fallback_min_confidence", 0.45);
        camera_fallback_ground_z_ = declare_parameter("camera_fallback_ground_z", 0.0);
        camera_fallback_pixel_y_ratio_ = declare_parameter("camera_fallback_pixel_y_ratio", 0.85);
        camera_fallback_min_x_ = declare_parameter("camera_fallback_min_x", 0.0);
        camera_fallback_max_x_ = declare_parameter("camera_fallback_max_x", 28.0);
        camera_fallback_min_y_ = declare_parameter("camera_fallback_min_y", -20.0);
        camera_fallback_max_y_ = declare_parameter("camera_fallback_max_y", 20.0);
        camera_fallback_max_range_ = declare_parameter("camera_fallback_max_range", 30.0);
        tracker_observation_noise_xy_ = declare_parameter("tracker_observation_noise_xy", 0.18);
        tracker_observation_noise_z_ = declare_parameter("tracker_observation_noise_z", 0.18);
        tracker_init_thresh_ = declare_parameter("tracker_init_thresh", 2);
        tracker_miss_thresh_ = declare_parameter("tracker_miss_thresh", 10);
        tracker_max_acceleration_ = declare_parameter("tracker_max_acceleration", 5.0);
        tracker_acceleration_correlation_time_ = declare_parameter("tracker_acceleration_correlation_time", 0.6);
        tracker_distance_weight_ = declare_parameter("tracker_distance_weight", 0.55);
        tracker_feature_weight_ = declare_parameter("tracker_feature_weight", 0.45);
        tracker_distance_thresh_ = declare_parameter("tracker_distance_thresh", 1.1);
        tracker_hard_match_distance_ = declare_parameter("tracker_hard_match_distance", 2.4);
        tracker_static_smooth_max_speed_ = declare_parameter("tracker_static_smooth_max_speed", 0.12);
        tracker_static_smooth_radius_ = declare_parameter("tracker_static_smooth_radius", 0.18);
        tracker_static_smooth_alpha_ = declare_parameter("tracker_static_smooth_alpha", 0.55);
        sync_queue_size_ = declare_parameter("sync_queue_size", 4);
        uav_detection_enabled_ = declare_parameter("uav_detection_enabled", false);
        visualization_enabled_ = declare_parameter("visualization_enabled", true);
        visualization_imshow_ = declare_parameter("visualization_imshow", true);
        visualization_window_width_ = declare_parameter("visualization_window_width", 1280);
        visualization_window_height_ = declare_parameter("visualization_window_height", 720);
        enemy_outpost_detection_enabled_ = declare_parameter("enemy_outpost_detection_enabled", false);
        enemy_outpost_x_ = declare_parameter("enemy_outpost_x", 17.581);
        enemy_outpost_y_ = declare_parameter("enemy_outpost_y", 11.902);
        enemy_outpost_z_min_ = declare_parameter("enemy_outpost_z_min", 5.71);
        enemy_outpost_z_max_ = declare_parameter("enemy_outpost_z_max", 11.10);
        enemy_outpost_bar_width_ = declare_parameter("enemy_outpost_bar_width", 24);
        enemy_outpost_alive_topic_ = declare_parameter("enemy_outpost_alive_topic", std::string("judge/enemy_outpost_alive"));
        enemy_outpost_min_light_pixels_ = declare_parameter("enemy_outpost_min_light_pixels", 30);
        enemy_outpost_min_light_ratio_ = declare_parameter("enemy_outpost_min_light_ratio", 0.001);
        enemy_outpost_min_saturation_ = declare_parameter("enemy_outpost_min_saturation", 80);
        enemy_outpost_min_value_ = declare_parameter("enemy_outpost_min_value", 160);
        locator_queue_size_ = declare_parameter("locator_queue_size", 2);
        locator_min_depth_diff_ = declare_parameter("locator_min_depth_diff", 500.0);
        locator_max_depth_diff_ = declare_parameter("locator_max_depth_diff", 4000.0);
        locator_cluster_tolerance_ = declare_parameter("locator_cluster_tolerance", 400.0);
        locator_min_cluster_size_ = declare_parameter("locator_min_cluster_size", 8);
        locator_max_cluster_size_ = declare_parameter("locator_max_cluster_size", 1000);
        locator_max_distance_ = declare_parameter("locator_max_distance", 29300.0);
        locator_adaptive_cluster_enabled_ = declare_parameter("locator_adaptive_cluster_enabled", false);
        locator_near_distance_ = declare_parameter("locator_near_distance", 10000.0);
        locator_mid_distance_ = declare_parameter("locator_mid_distance", 18000.0);
        locator_near_cluster_tolerance_ = declare_parameter("locator_near_cluster_tolerance", 400.0);
        locator_mid_cluster_tolerance_ = declare_parameter("locator_mid_cluster_tolerance", 550.0);
        locator_far_cluster_tolerance_ = declare_parameter("locator_far_cluster_tolerance", 750.0);
        locator_near_min_cluster_size_ = declare_parameter("locator_near_min_cluster_size", 8);
        locator_mid_min_cluster_size_ = declare_parameter("locator_mid_min_cluster_size", 6);
        locator_far_min_cluster_size_ = declare_parameter("locator_far_min_cluster_size", 4);
        locator_debug_enabled_ = declare_parameter("locator_debug_enabled", false);
        sentry_targets_enabled_ = declare_parameter("sentry_targets_enabled", true);
        sentry_targets_topic_ = declare_parameter("sentry_targets_topic", std::string("judge/sentry_targets"));
        sentry_targets_timeout_ms_ = declare_parameter("sentry_targets_timeout_ms", 500);
        sentry_targets_max_count_ = declare_parameter("sentry_targets_max_count", 6);
        sentry_targets_match_dist_ = declare_parameter("sentry_targets_match_dist", 0.5);
        fixed_slot_guess_enabled_ = declare_parameter("fixed_slot_guess_enabled", false);
        fixed_slot_guess_switch_interval_ms_ = declare_parameter("fixed_slot_guess_switch_interval_ms", 500);
        fixed_slot_guess_radius_m_ = declare_parameter("fixed_slot_guess_radius_m", 0.3);
        fixed_slot_guess_hero_point_ = declare_parameter<std::vector<double>>("fixed_slot_guess_hero_point", { 12.0, 14.2 });
        fixed_slot_guess_hero_point_side_ = declare_parameter("fixed_slot_guess_hero_point_side", std::string("own"));
        fixed_slot_guess_engineer_point_ = declare_parameter<std::vector<double>>("fixed_slot_guess_engineer_point", { 15.4, 7.3 });
        fixed_slot_guess_engineer_point_side_ = declare_parameter("fixed_slot_guess_engineer_point_side", std::string("enemy"));
        early_all_enemy_guess_enabled_ = declare_parameter("early_all_enemy_guess_enabled", false);
        early_all_enemy_guess_point_ = declare_parameter<std::vector<double>>("early_all_enemy_guess_point", { 10.8, 4.2 });
        early_all_enemy_guess_point_side_ = declare_parameter("early_all_enemy_guess_point_side", std::string("enemy"));
        early_all_enemy_guess_radius_m_ = declare_parameter("early_all_enemy_guess_radius_m", 0.3);
        early_all_enemy_guess_switch_interval_ms_ = declare_parameter("early_all_enemy_guess_switch_interval_ms", 500);
        early_all_enemy_guess_min_remain_time_sec_ = declare_parameter("early_all_enemy_guess_min_remain_time_sec", 240);
        early_all_enemy_guess_max_remain_time_sec_ = declare_parameter("early_all_enemy_guess_max_remain_time_sec", 420);
        late_all_enemy_guess_enabled_ = declare_parameter("late_all_enemy_guess_enabled", false);
        late_all_enemy_guess_points_ = declare_parameter<std::vector<double>>(
            "late_all_enemy_guess_points", { 12.1, 8.6, 12.0, 9.5, 10.7, 8.0 });
        late_all_enemy_guess_point_side_ = declare_parameter("late_all_enemy_guess_point_side", std::string("enemy"));
        late_all_enemy_guess_radius_m_ = declare_parameter("late_all_enemy_guess_radius_m", 0.2);
        late_all_enemy_guess_switch_interval_ms_ = declare_parameter("late_all_enemy_guess_switch_interval_ms", 500);
        late_all_enemy_guess_min_remain_time_sec_ = declare_parameter("late_all_enemy_guess_min_remain_time_sec", 0);
        late_all_enemy_guess_max_remain_time_sec_ = declare_parameter("late_all_enemy_guess_max_remain_time_sec", 240);
        std::transform(
            fixed_slot_guess_hero_point_side_.begin(), fixed_slot_guess_hero_point_side_.end(),
            fixed_slot_guess_hero_point_side_.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(
            fixed_slot_guess_engineer_point_side_.begin(), fixed_slot_guess_engineer_point_side_.end(),
            fixed_slot_guess_engineer_point_side_.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(
            early_all_enemy_guess_point_side_.begin(), early_all_enemy_guess_point_side_.end(),
            early_all_enemy_guess_point_side_.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(
            late_all_enemy_guess_point_side_.begin(), late_all_enemy_guess_point_side_.end(),
            late_all_enemy_guess_point_side_.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        auto default_team_color = declare_parameter("default_team_color", std::string("blue"));
        std::transform(default_team_color.begin(), default_team_color.end(), default_team_color.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        team_color_ = default_team_color == "red"
            ? radar_interface::team_color::C_RED
            : radar_interface::team_color::C_BLUE;

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, rclcpp::SystemDefaultsQoS(),
            std::bind(&RmRadarPipelineNode::camera_info_callback, this, std::placeholders::_1));
        team_color_sub_ = create_subscription<radar_interface::team_color::msg>(
            "judge/color", rclcpp::SystemDefaultsQoS(),
            std::bind(&RmRadarPipelineNode::team_color_callback, this, std::placeholders::_1));
        remain_time_sub_ = create_subscription<std_msgs::msg::UInt16>(
            "judge/remain_time", rclcpp::SystemDefaultsQoS(),
            std::bind(&RmRadarPipelineNode::remain_time_callback, this, std::placeholders::_1));
        if (sentry_targets_enabled_) {
            sentry_targets_sub_ = create_subscription<radar_interface::msg::TargetArray>(
                sentry_targets_topic_, rclcpp::SystemDefaultsQoS(),
                std::bind(&RmRadarPipelineNode::sentry_targets_callback, this, std::placeholders::_1));
        }

        image_sub_.subscribe(this, image_topic_, rmw_qos_profile_sensor_data);
        cloud_sub_.subscribe(this, pointcloud_topic_, rmw_qos_profile_sensor_data);
        sync_ = std::make_shared<Sync>(SyncPolicy(sync_queue_size_), image_sub_, cloud_sub_);
        sync_->registerCallback(std::bind(&RmRadarPipelineNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

        target_pub_ = create_publisher<radar_interface::msg::TargetArray>("rm_radar_pipeline/targets", rclcpp::SystemDefaultsQoS());
        detected_pub_ = create_publisher<radar_interface::msg::DetectedTargetArray>(
            "rm_radar_pipeline/detected_targets", rclcpp::SystemDefaultsQoS());
        match_pub_ = create_publisher<radar_interface::msg::MatchResult>("rm_radar_pipeline/match_result", rclcpp::SystemDefaultsQoS());
        if (uav_detection_enabled_)
            uav_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/radar/uav_target", rclcpp::SystemDefaultsQoS());
        if (enemy_outpost_detection_enabled_) {
            enemy_outpost_alive_pub_ = create_publisher<std_msgs::msg::Bool>(
                enemy_outpost_alive_topic_, rclcpp::QoS(rclcpp::KeepLast(10)));
        }
        if (visualization_enabled_)
            visualization_pub_ = create_publisher<sensor_msgs::msg::Image>("rm_radar_pipeline/visualization", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

        RCLCPP_INFO(get_logger(), "rm_radar_pipeline adapter started.");
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    struct CameraFallbackSlot {
        bool active = false;
        radar_interface::msg::Target target;
        int color = -1;
        int type = -1;
        rclcpp::Time last_seen {};
    };

    std::string default_model_path(const std::string& name) const
    {
        return ament_index_cpp::get_package_share_directory("rm_radar_pipeline") + "/models/" + name;
    }

    std::string default_nn_detector_model_path(const std::string& name) const
    {
        return ament_index_cpp::get_package_share_directory("nn_detector") + "/models/" + name;
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        std::scoped_lock lock(init_mutex_);
        if (camera_info_)
            return;
        camera_info_ = *msg;
    }

    void sentry_targets_callback(const radar_interface::msg::TargetArray::SharedPtr msg)
    {
        std::scoped_lock lock(sentry_targets_mutex_);
        latest_sentry_targets_ = *msg;
        latest_sentry_targets_time_ = now();
        has_sentry_targets_ = true;
    }

    void team_color_callback(const radar_interface::team_color::msg& msg)
    {
        const auto new_color = static_cast<radar_interface::team_color::ENUM>(msg.data);
        if (new_color != radar_interface::team_color::C_RED && new_color != radar_interface::team_color::C_BLUE)
            return;
        if (team_color_ != new_color) {
            team_color_ = new_color;
            fixed_slot_guess_state_ = {};
            early_all_enemy_guess_state_ = {};
            late_all_enemy_guess_state_ = {};
        }
    }

    void remain_time_callback(const std_msgs::msg::UInt16& msg)
    {
        latest_remain_time_sec_ = msg.data;
        has_remain_time_ = true;
    }

    std::vector<radar_interface::msg::Target> get_recent_sentry_targets()
    {
        std::scoped_lock lock(sentry_targets_mutex_);
        if (!sentry_targets_enabled_ || !has_sentry_targets_)
            return {};

        if (sentry_targets_timeout_ms_ >= 0 &&
            (now() - latest_sentry_targets_time_).nanoseconds() >
                static_cast<int64_t>(sentry_targets_timeout_ms_) * 1000 * 1000) {
            return {};
        }

        std::vector<radar_interface::msg::Target> targets;
        const size_t limit = static_cast<size_t>(std::max(0, sentry_targets_max_count_));
        targets.reserve(std::min(limit, latest_sentry_targets_.targets.size()));
        for (const auto& target : latest_sentry_targets_.targets) {
            if (targets.size() >= limit)
                break;
            if (!std::isfinite(target.position[0]) || !std::isfinite(target.position[1]) ||
                !std::isfinite(target.calc_z)) {
                continue;
            }
            targets.push_back(target);
        }
        return targets;
    }

    bool ensure_pipeline()
    {
        std::scoped_lock lock(init_mutex_);
        if (!camera_info_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera_info on %s", camera_info_topic_.c_str());
            return false;
        }

        geometry_msgs::msg::TransformStamped lidar_to_camera;
        geometry_msgs::msg::TransformStamped world_to_camera;
        try {
            lidar_to_camera = tf_buffer_.lookupTransform(camera_frame_, lidar_frame_, tf2::TimePointZero);
            world_to_camera = tf_buffer_.lookupTransform(camera_frame_, world_frame_, tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for TF: %s", ex.what());
            return false;
        }

        const Eigen::Matrix4d lidar_to_camera_mat = tf2::transformToEigen(lidar_to_camera).matrix();
        const Eigen::Matrix4d world_to_camera_mat = tf2::transformToEigen(world_to_camera).matrix();
        const bool tf_updated = !locator_tf_ready_
            || transform_changed(last_lidar_to_camera_, lidar_to_camera_mat)
            || transform_changed(last_world_to_camera_, world_to_camera_mat);

        if (detector_ && locator_ && tracker_ && !tf_updated)
            return true;

        cv::Matx33f intrinsic(
            static_cast<float>(camera_info_->k[0]), static_cast<float>(camera_info_->k[1]), static_cast<float>(camera_info_->k[2]),
            static_cast<float>(camera_info_->k[3]), static_cast<float>(camera_info_->k[4]), static_cast<float>(camera_info_->k[5]),
            static_cast<float>(camera_info_->k[6]), static_cast<float>(camera_info_->k[7]), static_cast<float>(camera_info_->k[8]));
        const cv::Size image_size(static_cast<int>(camera_info_->width), static_cast<int>(camera_info_->height));

        constexpr float kMetersToMillimeters = 1000.0f;
        if (!detector_) {
            detector_ = std::make_unique<CvDnnRobotDetector>(
                car_model_path_, armor_model_path_, cv::Size(640, 640), kClassNum,
                armor_class_color_map_, armor_class_type_map_, car_class_num_, car_class_filter_,
                static_cast<float>(car_confidence_threshold_), static_cast<float>(armor_confidence_threshold_),
                static_cast<float>(nms_threshold_));
        }
        locator_ = std::make_unique<radar::Locator>(
            image_size.width, image_size.height, intrinsic,
            to_scaled_mat(lidar_to_camera, kMetersToMillimeters),
            to_scaled_mat(world_to_camera, kMetersToMillimeters),
            0.5f,
            static_cast<size_t>(locator_queue_size_),
            static_cast<float>(locator_min_depth_diff_),
            static_cast<float>(locator_max_depth_diff_),
            static_cast<float>(locator_cluster_tolerance_),
            locator_min_cluster_size_,
            locator_max_cluster_size_,
            static_cast<float>(locator_max_distance_),
            locator_adaptive_cluster_enabled_,
            static_cast<float>(locator_near_distance_),
            static_cast<float>(locator_mid_distance_),
            static_cast<float>(locator_near_cluster_tolerance_),
            static_cast<float>(locator_mid_cluster_tolerance_),
            static_cast<float>(locator_far_cluster_tolerance_),
            locator_near_min_cluster_size_,
            locator_mid_min_cluster_size_,
            locator_far_min_cluster_size_);
        tracker_ = std::make_unique<radar::Tracker>(
            cv::Point3f(
                static_cast<float>(tracker_observation_noise_xy_),
                static_cast<float>(tracker_observation_noise_xy_),
                static_cast<float>(tracker_observation_noise_z_)),
            kClassNum,
            tracker_init_thresh_,
            tracker_miss_thresh_,
            static_cast<float>(tracker_max_acceleration_),
            static_cast<float>(tracker_acceleration_correlation_time_),
            static_cast<float>(tracker_distance_weight_),
            static_cast<float>(tracker_feature_weight_),
            100,
            static_cast<float>(tracker_distance_thresh_),
            static_cast<float>(tracker_hard_match_distance_),
            static_cast<float>(tracker_static_smooth_max_speed_),
            static_cast<float>(tracker_static_smooth_radius_),
            static_cast<float>(tracker_static_smooth_alpha_));
        last_lidar_to_camera_ = lidar_to_camera_mat;
        last_world_to_camera_ = world_to_camera_mat;
        locator_tf_ready_ = true;
        RCLCPP_INFO(get_logger(), "Imported rm_radar locator/tracker %s.", tf_updated ? "refreshed after TF update" : "initialized");
        return true;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_mm(const sensor_msgs::msg::PointCloud2& msg) const
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::PointCloud<pcl::PointXYZ> meters;
        pcl::fromROSMsg(msg, meters);
        cloud->reserve(meters.size());
        for (const auto& p : meters) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            cloud->emplace_back(p.x * 1000.0f, p.y * 1000.0f, p.z * 1000.0f);
        }
        return cloud;
    }


    cv::Mat image_msg_to_bgr(const sensor_msgs::msg::Image& msg) const
    {
        int channels = 0;
        if (msg.encoding == "bgr8" || msg.encoding == "rgb8") {
            channels = 3;
        } else if (msg.encoding == "mono8") {
            channels = 1;
        } else {
            throw std::runtime_error("unsupported encoding: " + msg.encoding);
        }

        if (msg.height == 0 || msg.width == 0 || msg.step < msg.width * channels) {
            throw std::runtime_error("invalid image shape");
        }
        const size_t required = static_cast<size_t>(msg.step) * msg.height;
        if (msg.data.size() < required) {
            throw std::runtime_error("image data is shorter than step * height");
        }

        cv::Mat view(static_cast<int>(msg.height), static_cast<int>(msg.width),
            channels == 3 ? CV_8UC3 : CV_8UC1, const_cast<unsigned char*>(msg.data.data()), msg.step);
        cv::Mat bgr;
        if (msg.encoding == "rgb8") {
            cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
        } else if (msg.encoding == "mono8") {
            cv::cvtColor(view, bgr, cv::COLOR_GRAY2BGR);
        } else {
            bgr = view.clone();
        }
        return bgr;
    }

    sensor_msgs::msg::Image mat_to_image_msg(const std_msgs::msg::Header& header, const cv::Mat& image) const
    {
        if (image.empty() || image.type() != CV_8UC3) {
            throw std::runtime_error("visualization image must be non-empty bgr8");
        }
        cv::Mat contiguous = image.isContinuous() ? image : image.clone();
        sensor_msgs::msg::Image msg;
        msg.header = header;
        msg.height = static_cast<uint32_t>(contiguous.rows);
        msg.width = static_cast<uint32_t>(contiguous.cols);
        msg.encoding = "bgr8";
        msg.is_bigendian = false;
        msg.step = static_cast<uint32_t>(contiguous.cols * contiguous.elemSize());
        msg.data.assign(contiguous.datastart, contiguous.dataend);
        return msg;
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg)
    {
        if (!ensure_pipeline())
            return;

        cv::Mat image;
        try {
            image = image_msg_to_bgr(*image_msg);
        } catch (const std::exception& ex) {
            RCLCPP_WARN(get_logger(), "image conversion failed: %s", ex.what());
            return;
        }

        auto cloud = cloud_to_mm(*cloud_msg);
        publish_uav_target_if_enabled(*cloud_msg);
        publish_enemy_outpost_alive_if_enabled(image);
        locator_->update(cloud);
        locator_->cluster();
        auto robots = detector_->detect(image);
        locator_->search(robots);
        log_locator_debug(robots);
        tracker_->update(robots, to_chrono_time(image_msg->header.stamp));
        publish_visualization_if_enabled(image_msg->header, image, robots);
        publish_outputs(image_msg->header, robots);
    }

    void log_locator_debug(const std::vector<radar::Robot>& robots)
    {
        if (!locator_debug_enabled_ || !locator_)
            return;

        size_t located_count = 0;
        for (const auto& robot : robots) {
            if (robot.location())
                ++located_count;
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "locator debug: detections=%zu located=%zu input_points=%zu zero_points=%zu max_dist_rejects=%zu invalid_camera_rejects=%zu image_bounds_rejects=%zu projected_points=%zu depth_pixels=%zu foreground_points=%zu clusters=%zu roi_candidate_points=%zu",
            robots.size(),
            located_count,
            locator_->lastInputPointCount(),
            locator_->lastZeroPointCount(),
            locator_->lastMaxDistanceRejectCount(),
            locator_->lastInvalidCameraRejectCount(),
            locator_->lastImageBoundsRejectCount(),
            locator_->lastProjectedPointCount(),
            locator_->lastDepthPixelCount(),
            locator_->foregroundPointCount(),
            locator_->clusterCount(),
            locator_->lastSearchCandidatePointCount());
    }

    std::optional<cv::Point2d> project_world_to_image(double x, double y, double z) const
    {
        if (!camera_info_)
            return std::nullopt;

        Eigen::Vector4d world_point(x, y, z, 1.0);
        const Eigen::Vector4d camera_point = last_world_to_camera_ * world_point;
        if (!camera_point.allFinite() || camera_point.z() <= 1e-6)
            return std::nullopt;

        double xn = camera_point.x() / camera_point.z();
        double yn = camera_point.y() / camera_point.z();

        const auto& d = camera_info_->d;
        if (camera_info_->distortion_model == "plumb_bob" && d.size() >= 5) {
            const double k1 = d[0];
            const double k2 = d[1];
            const double p1 = d[2];
            const double p2 = d[3];
            const double k3 = d[4];
            const double r2 = xn * xn + yn * yn;
            const double r4 = r2 * r2;
            const double r6 = r4 * r2;
            const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            const double x_distorted = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
            const double y_distorted = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
            xn = x_distorted;
            yn = y_distorted;
        }

        const double u = camera_info_->k[0] * xn + camera_info_->k[2];
        const double v = camera_info_->k[4] * yn + camera_info_->k[5];
        if (!std::isfinite(u) || !std::isfinite(v))
            return std::nullopt;
        return cv::Point2d(u, v);
    }

    void publish_enemy_outpost_alive_if_enabled(const cv::Mat& image)
    {
        if (!enemy_outpost_detection_enabled_ || !enemy_outpost_alive_pub_ || image.empty())
            return;

        const double z_min = std::min(enemy_outpost_z_min_, enemy_outpost_z_max_);
        const double z_max = std::max(enemy_outpost_z_min_, enemy_outpost_z_max_);
        const auto lower_pixel = project_world_to_image(enemy_outpost_x_, enemy_outpost_y_, z_min);
        const auto upper_pixel = project_world_to_image(enemy_outpost_x_, enemy_outpost_y_, z_max);
        bool enemy_outpost_alive = false;
        const auto expected_color = enemy_color_for(team_color_);
        if (lower_pixel && upper_pixel) {
            cv::Mat roi_mask = cv::Mat::zeros(image.size(), CV_8UC1);
            cv::line(
                roi_mask,
                cv::Point(cvRound(lower_pixel->x), cvRound(lower_pixel->y)),
                cv::Point(cvRound(upper_pixel->x), cvRound(upper_pixel->y)),
                cv::Scalar(255),
                std::max(1, enemy_outpost_bar_width_),
                cv::LINE_AA);
            enemy_outpost_alive = has_expected_light(
                image, roi_mask, expected_color,
                enemy_outpost_min_light_pixels_,
                enemy_outpost_min_light_ratio_,
                enemy_outpost_min_saturation_,
                enemy_outpost_min_value_);
        }

        std_msgs::msg::Bool alive_msg;
        alive_msg.data = enemy_outpost_alive;
        enemy_outpost_alive_pub_->publish(alive_msg);

        if (!has_enemy_outpost_alive_ || enemy_outpost_alive != last_enemy_outpost_alive_) {
            RCLCPP_INFO(
                get_logger(),
                "Enemy outpost light state: %s expected_color=%s at world=(%.3f,%.3f,z %.3f..%.3f)",
                enemy_outpost_alive ? "alive" : "destroyed",
                color_name(expected_color),
                enemy_outpost_x_,
                enemy_outpost_y_,
                z_min,
                z_max);
            has_enemy_outpost_alive_ = true;
            last_enemy_outpost_alive_ = enemy_outpost_alive;
        }
    }

    void publish_visualization_if_enabled(const std_msgs::msg::Header& header, const cv::Mat& image,
        const std::vector<radar::Robot>& robots)
    {
        if (!visualization_enabled_ || image.empty())
            return;

        cv::Mat canvas = image.clone();
        for (const auto& robot : robots) {
            if (!robot.rect())
                continue;

            cv::Rect car_rect = robot.rect().value();
            car_rect &= cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (car_rect.empty())
                continue;

            const bool is_red = robot.label() && robot.label().value() >= radar::Label::RedHero && robot.label().value() <= radar::Label::RedSentry;
            const cv::Scalar car_color = is_red ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 80, 0);
            cv::rectangle(canvas, car_rect, car_color, 2);

            std::string text = robot.label() ? label_name(robot.label().value()) : "CAR";
            if (robot.confidence()) {
                text += cv::format(" %.2f", robot.confidence().value());
            }
            if (robot.track_id()) {
                text += cv::format(" #%d", robot.track_id().value());
            }
            const int text_y = std::max(20, car_rect.y - 6);
            cv::putText(canvas, text, cv::Point(car_rect.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.65, car_color, 2);

            if (robot.armors()) {
                for (const auto& armor : robot.armors().value()) {
                    cv::Rect armor_rect(cvRound(armor.x), cvRound(armor.y), cvRound(armor.width), cvRound(armor.height));
                    armor_rect &= cv::Rect(0, 0, canvas.cols, canvas.rows);
                    if (armor_rect.empty())
                        continue;
                    cv::rectangle(canvas, armor_rect, cv::Scalar(0, 255, 255), 2);
                    cv::putText(canvas, label_name(static_cast<int>(armor.label)),
                        cv::Point(armor_rect.x, std::max(16, armor_rect.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
                }
            }

            if (robot.location()) {
                const auto loc = robot.location().value();
                cv::putText(canvas, cv::format("(%.2f, %.2f, %.2f)m", loc.x, loc.y, loc.z),
                    cv::Point(car_rect.x, std::min(canvas.rows - 8, car_rect.y + car_rect.height + 18)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 255, 80), 1);
            }
        }

        if (visualization_pub_) {
            try {
                visualization_pub_->publish(mat_to_image_msg(header, canvas));
            } catch (const std::exception& ex) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "visualization publish failed: %s", ex.what());
            }
        }
        if (visualization_imshow_) {
            if (!visualization_window_initialized_) {
                cv::namedWindow("rm_radar_pipeline", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                if (visualization_window_width_ > 0 && visualization_window_height_ > 0)
                    cv::resizeWindow("rm_radar_pipeline", visualization_window_width_, visualization_window_height_);
                visualization_window_initialized_ = true;
            }
            cv::imshow("rm_radar_pipeline", canvas);
            cv::waitKey(1);
        }
    }

    radar_interface::msg::Target make_target(const radar::Robot& robot) const
    {
        radar_interface::msg::Target target;
        target.id = static_cast<uint64_t>(robot.track_id().value_or(next_fallback_id_++));
        const auto loc = robot.location().value();
        target.position = { static_cast<double>(loc.x), static_cast<double>(loc.y) };
        target.pos_covariance = { 0.05, 0.0, 0.0, 0.05 };
        if (robot.velocity()) {
            const auto vel = robot.velocity().value();
            target.velocity = { static_cast<double>(vel.x), static_cast<double>(vel.y) };
        } else {
            target.velocity = { 0.0, 0.0 };
        }
        target.vel_covariance = { 1.0, 0.0, 0.0, 1.0 };
        target.calc_z = loc.z;
        target.observed_pos = { static_cast<double>(loc.x), static_cast<double>(loc.y), static_cast<double>(loc.z) };
        target.uncertainty = robot.track_state().has_value() && robot.track_state().value() == radar::TrackState::Confirmed ? 0 : 1;
        return target;
    }

    int camera_fallback_index(int color, int type) const
    {
        const int slot = type_to_slot(type);
        if (slot < 0)
            return -1;
        if (color == radar_interface::msg::Armor::COLOR_BLUE)
            return slot;
        if (color == radar_interface::msg::Armor::COLOR_RED)
            return kRobotsPerTeam + slot;
        return -1;
    }

    radar_interface::msg::MatchedTarget& match_slot_for(
        radar_interface::msg::MatchResult& match_msg, int color, int type)
    {
        const int slot = type_to_slot(type);
        auto& slots = color == radar_interface::msg::Armor::COLOR_BLUE ? match_msg.blue : match_msg.red;
        return slots[slot];
    }

    std::optional<cv::Point3d> estimate_camera_fallback_position(const radar::Robot& robot) const
    {
        if (!camera_info_ || !robot.rect())
            return std::nullopt;

        const auto rect = robot.rect().value();
        const double fx = camera_info_->k[0];
        const double fy = camera_info_->k[4];
        const double cx = camera_info_->k[2];
        const double cy = camera_info_->k[5];
        if (std::abs(fx) < 1e-6 || std::abs(fy) < 1e-6)
            return std::nullopt;

        const double pixel_x = rect.x + rect.width * 0.5;
        const double pixel_y = rect.y + rect.height * std::clamp(camera_fallback_pixel_y_ratio_, 0.0, 1.0);
        const Eigen::Vector3d ray_camera((pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0);
        const Eigen::Matrix4d camera_to_world = last_world_to_camera_.inverse();
        const Eigen::Vector3d origin = camera_to_world.block<3, 1>(0, 3);
        Eigen::Vector3d direction = camera_to_world.block<3, 3>(0, 0) * ray_camera;
        if (!direction.allFinite() || std::abs(direction.z()) < 1e-6)
            return std::nullopt;

        const double scale = (camera_fallback_ground_z_ - origin.z()) / direction.z();
        if (!std::isfinite(scale) || scale <= 0.0 || scale > camera_fallback_max_range_)
            return std::nullopt;

        const Eigen::Vector3d world = origin + direction * scale;
        if (!world.allFinite())
            return std::nullopt;
        if (world.x() < camera_fallback_min_x_ || world.x() > camera_fallback_max_x_ ||
            world.y() < camera_fallback_min_y_ || world.y() > camera_fallback_max_y_) {
            return std::nullopt;
        }

        return cv::Point3d(world.x(), world.y(), camera_fallback_ground_z_);
    }

    radar_interface::msg::Target make_camera_fallback_target(
        int fallback_index, const cv::Point3d& position) const
    {
        radar_interface::msg::Target target;
        target.id = static_cast<uint64_t>(2000000 + fallback_index);
        target.position = { position.x, position.y };
        target.pos_covariance = { 0.8, 0.0, 0.0, 0.8 };
        target.velocity = { 0.0, 0.0 };
        target.vel_covariance = { 4.0, 0.0, 0.0, 4.0 };
        target.calc_z = position.z;
        target.observed_pos = { position.x, position.y, position.z };
        target.uncertainty = 0;
        return target;
    }

    void update_camera_fallbacks(
        const std::vector<radar::Robot>& robots,
        radar_interface::msg::TargetArray& targets_msg,
        radar_interface::msg::DetectedTargetArray& detected_msg,
        radar_interface::msg::MatchResult& match_msg)
    {
        if (!camera_fallback_enabled_ || !locator_tf_ready_)
            return;

        const auto now_time = now();
        for (const auto& robot : robots) {
            if (!robot.label() || !robot.rect())
                continue;
            if (robot.confidence().value_or(0.0f) < camera_fallback_min_confidence_)
                continue;

            const auto color_type = label_to_color_type(robot.label().value());
            if (!color_type)
                continue;
            const int fallback_idx = camera_fallback_index(color_type->first, color_type->second);
            if (fallback_idx < 0 || fallback_idx >= static_cast<int>(camera_fallback_slots_.size()))
                continue;

            auto& matched = match_slot_for(match_msg, color_type->first, color_type->second);
            if (matched.id != -1) {
                camera_fallback_slots_[fallback_idx].active = false;
                continue;
            }

            const auto position = estimate_camera_fallback_position(robot);
            if (!position)
                continue;

            auto& slot = camera_fallback_slots_[fallback_idx];
            slot.active = true;
            slot.color = color_type->first;
            slot.type = color_type->second;
            slot.target = make_camera_fallback_target(fallback_idx, position.value());
            slot.last_seen = now_time;
        }

        const int64_t hold_ns = static_cast<int64_t>(std::max(0, camera_fallback_hold_ms_)) * 1000000LL;
        for (auto& slot : camera_fallback_slots_) {
            if (!slot.active)
                continue;
            if (hold_ns <= 0 || slot.last_seen.nanoseconds() == 0 ||
                (now_time - slot.last_seen).nanoseconds() > hold_ns) {
                slot.active = false;
                continue;
            }

            auto& matched = match_slot_for(match_msg, slot.color, slot.type);
            if (matched.id != -1 && matched.id != static_cast<int64_t>(slot.target.id)) {
                slot.active = false;
                continue;
            }

            targets_msg.targets.push_back(slot.target);
            radar_interface::msg::DetectedTarget detected;
            detected.target = slot.target;
            detected.color = static_cast<int8_t>(slot.color);
            detected.type = static_cast<int8_t>(slot.type);
            detected_msg.targets.push_back(detected);
            matched.id = static_cast<int64_t>(slot.target.id);
            matched.position = slot.target.position;
        }
    }

    std::optional<cv::Point2d> fixed_slot_reference_to_enemy_point(
        const std::vector<double>& reference_point,
        const std::string& point_side) const
    {
        if (reference_point.size() < 2 || !std::isfinite(reference_point[0]) || !std::isfinite(reference_point[1]))
            return std::nullopt;

        const cv::Point2d point(reference_point[0], reference_point[1]);
        bool mirror = team_color_ == radar_interface::team_color::C_BLUE;
        if (point_side == "own")
            mirror = team_color_ == radar_interface::team_color::C_RED;
        else if (point_side == "enemy")
            mirror = team_color_ == radar_interface::team_color::C_BLUE;
        else if (point_side == "red")
            mirror = team_color_ == radar_interface::team_color::C_RED;
        else if (point_side == "blue")
            mirror = team_color_ == radar_interface::team_color::C_BLUE;
        else if (point_side == "absolute")
            mirror = false;

        if (!mirror)
            return point;
        return cv::Point2d(kFieldWidthM - point.x, kFieldHeightM - point.y);
    }

    cv::Point2d guess_candidate(const cv::Point2d& center, size_t index, double radius) const
    {
        static constexpr std::array<std::array<double, 2>, 9> kOffsets = {
            std::array<double, 2> { 0.0, 0.0 },
            std::array<double, 2> { 1.0, 0.0 },
            std::array<double, 2> { 0.0, 1.0 },
            std::array<double, 2> { -1.0, 0.0 },
            std::array<double, 2> { 0.0, -1.0 },
            std::array<double, 2> { 0.7071, 0.7071 },
            std::array<double, 2> { -0.7071, 0.7071 },
            std::array<double, 2> { -0.7071, -0.7071 },
            std::array<double, 2> { 0.7071, -0.7071 },
        };
        const auto& offset = kOffsets[index % kOffsets.size()];
        return cv::Point2d(center.x + offset[0] * radius,
            center.y + offset[1] * radius);
    }

    void apply_fixed_slot_guess(
        radar_interface::msg::MatchedTarget& slot,
        int64_t guess_id,
        const std::vector<double>& reference_point,
        const std::string& point_side,
        size_t candidate_index)
    {
        if (slot.id != -1)
            return;

        const auto enemy_center = fixed_slot_reference_to_enemy_point(reference_point, point_side);
        if (!enemy_center)
            return;

        const auto point = guess_candidate(enemy_center.value(), candidate_index, fixed_slot_guess_radius_m_);
        slot.id = guess_id;
        slot.position = { point.x, point.y };
    }

    void apply_fixed_slot_guesses(radar_interface::msg::MatchResult& match_msg)
    {
        if (!fixed_slot_guess_enabled_)
            return;

        const int interval_ms = std::max(1, fixed_slot_guess_switch_interval_ms_);
        const auto now_time = now();
        if (fixed_slot_guess_state_.last_switch_time.nanoseconds() == 0) {
            fixed_slot_guess_state_.last_switch_time = now_time;
        } else if ((now_time - fixed_slot_guess_state_.last_switch_time).nanoseconds() >=
            static_cast<int64_t>(interval_ms) * 1000 * 1000) {
            ++fixed_slot_guess_state_.candidate_index;
            fixed_slot_guess_state_.last_switch_time = now_time;
        }

        auto& enemy_slots = team_color_ == radar_interface::team_color::C_BLUE ? match_msg.red : match_msg.blue;
        apply_fixed_slot_guess(
            enemy_slots[radar_interface::msg::Armor::TYPE_HERO], -71,
            fixed_slot_guess_hero_point_, fixed_slot_guess_hero_point_side_,
            fixed_slot_guess_state_.candidate_index);
        apply_fixed_slot_guess(
            enemy_slots[radar_interface::msg::Armor::TYPE_ENGINEER], -72,
            fixed_slot_guess_engineer_point_, fixed_slot_guess_engineer_point_side_,
            fixed_slot_guess_state_.candidate_index + 3);
    }

    bool early_all_enemy_guess_time_active() const
    {
        if (!has_remain_time_)
            return false;
        const uint16_t min_remain = static_cast<uint16_t>(std::max(0, early_all_enemy_guess_min_remain_time_sec_));
        const uint16_t max_remain = static_cast<uint16_t>(std::max(0, early_all_enemy_guess_max_remain_time_sec_));
        return latest_remain_time_sec_ >= min_remain && latest_remain_time_sec_ <= max_remain;
    }

    bool late_all_enemy_guess_time_active() const
    {
        if (!has_remain_time_)
            return false;
        const uint16_t min_remain = static_cast<uint16_t>(std::max(0, late_all_enemy_guess_min_remain_time_sec_));
        const uint16_t max_remain = static_cast<uint16_t>(std::max(0, late_all_enemy_guess_max_remain_time_sec_));
        return latest_remain_time_sec_ >= min_remain && latest_remain_time_sec_ <= max_remain;
    }

    std::optional<cv::Point2d> point_from_flat_pairs(const std::vector<double>& points, size_t pair_index) const
    {
        const size_t pair_count = points.size() / 2;
        if (pair_count == 0)
            return std::nullopt;
        const size_t offset = (pair_index % pair_count) * 2;
        if (!std::isfinite(points[offset]) || !std::isfinite(points[offset + 1]))
            return std::nullopt;
        return cv::Point2d(points[offset], points[offset + 1]);
    }

    void apply_early_all_enemy_guesses(radar_interface::msg::MatchResult& match_msg)
    {
        if (!early_all_enemy_guess_enabled_ || !early_all_enemy_guess_time_active())
            return;

        const auto enemy_center = fixed_slot_reference_to_enemy_point(
            early_all_enemy_guess_point_, early_all_enemy_guess_point_side_);
        if (!enemy_center)
            return;

        const int interval_ms = std::max(1, early_all_enemy_guess_switch_interval_ms_);
        const auto now_time = now();
        if (early_all_enemy_guess_state_.last_switch_time.nanoseconds() == 0) {
            early_all_enemy_guess_state_.last_switch_time = now_time;
        } else if ((now_time - early_all_enemy_guess_state_.last_switch_time).nanoseconds() >=
            static_cast<int64_t>(interval_ms) * 1000 * 1000) {
            ++early_all_enemy_guess_state_.candidate_index;
            early_all_enemy_guess_state_.last_switch_time = now_time;
        }

        auto& enemy_slots = team_color_ == radar_interface::team_color::C_BLUE ? match_msg.red : match_msg.blue;
        for (size_t slot = 0; slot < enemy_slots.size(); ++slot) {
            auto& target = enemy_slots[slot];
            if (target.id != -1)
                continue;
            const auto point = guess_candidate(
                enemy_center.value(),
                early_all_enemy_guess_state_.candidate_index + slot,
                early_all_enemy_guess_radius_m_);
            target.id = -800 - static_cast<int64_t>(slot);
            target.position = { point.x, point.y };
        }
    }

    void apply_late_all_enemy_guesses(radar_interface::msg::MatchResult& match_msg)
    {
        if (!late_all_enemy_guess_enabled_ || !late_all_enemy_guess_time_active())
            return;

        const int interval_ms = std::max(1, late_all_enemy_guess_switch_interval_ms_);
        const auto now_time = now();
        if (late_all_enemy_guess_state_.last_switch_time.nanoseconds() == 0) {
            late_all_enemy_guess_state_.last_switch_time = now_time;
        } else if ((now_time - late_all_enemy_guess_state_.last_switch_time).nanoseconds() >=
            static_cast<int64_t>(interval_ms) * 1000 * 1000) {
            ++late_all_enemy_guess_state_.candidate_index;
            late_all_enemy_guess_state_.last_switch_time = now_time;
        }

        auto& enemy_slots = team_color_ == radar_interface::team_color::C_BLUE ? match_msg.red : match_msg.blue;
        for (size_t slot = 0; slot < enemy_slots.size(); ++slot) {
            auto& target = enemy_slots[slot];
            if (target.id != -1)
                continue;

            const auto center = point_from_flat_pairs(
                late_all_enemy_guess_points_,
                late_all_enemy_guess_state_.candidate_index + slot);
            if (!center)
                continue;
            const std::vector<double> center_vec { center->x, center->y };
            const auto enemy_center = fixed_slot_reference_to_enemy_point(
                center_vec, late_all_enemy_guess_point_side_);
            if (!enemy_center)
                continue;

            const auto point = guess_candidate(
                enemy_center.value(),
                late_all_enemy_guess_state_.candidate_index + slot,
                late_all_enemy_guess_radius_m_);
            target.id = -900 - static_cast<int64_t>(slot);
            target.position = { point.x, point.y };
        }
    }

    bool apply_sentry_position_override(
        radar_interface::msg::Target& target,
        const std::vector<radar_interface::msg::Target>& sentry_targets,
        std::vector<bool>& sentry_used) const
    {
        if (sentry_targets.empty() || sentry_targets_match_dist_ <= 0.0)
            return false;

        const double max_dist2 = sentry_targets_match_dist_ * sentry_targets_match_dist_;
        int best_idx = -1;
        double best_dist2 = max_dist2;
        for (size_t i = 0; i < sentry_targets.size(); ++i) {
            if (i < sentry_used.size() && sentry_used[i])
                continue;

            const auto& sentry_target = sentry_targets[i];
            if (!std::isfinite(sentry_target.position[0]) ||
                !std::isfinite(sentry_target.position[1]))
                continue;

            const double dx = target.position[0] - sentry_target.position[0];
            const double dy = target.position[1] - sentry_target.position[1];
            const double dist2 = dx * dx + dy * dy;
            if (dist2 <= best_dist2) {
                best_dist2 = dist2;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx < 0)
            return false;

        const auto& sentry_target = sentry_targets[best_idx];
        target.position[0] = sentry_target.position[0];
        target.position[1] = sentry_target.position[1];
        if (std::isfinite(sentry_target.calc_z))
            target.calc_z = sentry_target.calc_z;
        target.observed_pos = { target.position[0], target.position[1], target.calc_z };
        if (static_cast<size_t>(best_idx) < sentry_used.size())
            sentry_used[best_idx] = true;
        return true;
    }

    void publish_outputs(const std_msgs::msg::Header& header, const std::vector<radar::Robot>& robots)
    {
        radar_interface::msg::TargetArray targets_msg;
        radar_interface::msg::DetectedTargetArray detected_msg;
        radar_interface::msg::MatchResult match_msg;
        targets_msg.header = header;
        detected_msg.header = header;
        for (auto& t : match_msg.blue)
            t.id = -1;
        for (auto& t : match_msg.red)
            t.id = -1;

        const auto sentry_targets = get_recent_sentry_targets();
        std::vector<bool> sentry_used(sentry_targets.size(), false);
        int sentry_overrides = 0;
        for (const auto& robot : robots) {
            if (!robot.location())
                continue;
            if (!publish_tentative_tracks_ && (!robot.track_state() || robot.track_state().value() != radar::TrackState::Confirmed))
                continue;
            if (robot.confidence().value_or(1.0f) < min_publish_confidence_)
                continue;

            auto target = make_target(robot);
            if (apply_sentry_position_override(target, sentry_targets, sentry_used))
                ++sentry_overrides;
            targets_msg.targets.push_back(target);

            if (robot.label()) {
                auto color_type = label_to_color_type(robot.label().value());
                if (color_type) {
                    radar_interface::msg::DetectedTarget detected;
                    detected.target = target;
                    detected.color = static_cast<int8_t>(color_type->first);
                    detected.type = static_cast<int8_t>(color_type->second);
                    detected_msg.targets.push_back(detected);

                    auto& slots = color_type->first == radar_interface::msg::Armor::COLOR_BLUE ? match_msg.blue : match_msg.red;
                    const int slot = type_to_slot(color_type->second);
                    if (slot >= 0 && slot < static_cast<int>(slots.size())) {
                        slots[slot].id = static_cast<int64_t>(target.id);
                        slots[slot].position = target.position;
                    }
                }
            }
        }

        update_camera_fallbacks(robots, targets_msg, detected_msg, match_msg);
        apply_fixed_slot_guesses(match_msg);
        apply_early_all_enemy_guesses(match_msg);
        apply_late_all_enemy_guesses(match_msg);
        if (sentry_overrides > 0) {
            RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "Applied %d sentry position overrides within %.2fm.",
                sentry_overrides, sentry_targets_match_dist_);
        }

        target_pub_->publish(targets_msg);
        detected_pub_->publish(detected_msg);
        match_pub_->publish(match_msg);
    }

    void publish_uav_target_if_enabled(const sensor_msgs::msg::PointCloud2& cloud_msg)
    {
        if (!uav_detection_enabled_ || !uav_pub_)
            return;

        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(cloud_msg, cloud);
        if (cloud.empty())
            return;

        if (cloud_msg.header.frame_id != lidar_frame_) {
            try {
                const auto tf_msg = tf_buffer_.lookupTransform(lidar_frame_, cloud_msg.header.frame_id, tf2::TimePointZero);
                const Eigen::Affine3d tf = tf2::transformToEigen(tf_msg);
                if (!tf.matrix().allFinite()) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid TF for UAV detection, skip this frame.");
                    return;
                }
                pcl::PointCloud<pcl::PointXYZ> transformed;
                pcl::transformPointCloud(cloud, transformed, tf);
                cloud = std::move(transformed);
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for UAV TF: %s", ex.what());
                return;
            }
        }

        pcl::PointCloud<pcl::PointXYZ> fly_cloud;
        for (const auto& point : cloud.points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                continue;
            if (point.x > 3.0f && point.x < 27.5f
                && point.y > 0.2f && point.y < 10.0f
                && point.z > 1.7f && point.z < 3.0f) {
                fly_cloud.push_back(point);
            }
        }

        if (fly_cloud.size() <= 20)
            return;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(fly_cloud, centroid);
        geometry_msgs::msg::PointStamped msg;
        msg.header = cloud_msg.header;
        msg.header.frame_id = lidar_frame_;
        msg.point.x = centroid[0];
        msg.point.y = centroid[1];
        msg.point.z = centroid[2];
        uav_pub_->publish(msg);
    }

    std::string image_topic_;
    std::string camera_info_topic_;
    std::string pointcloud_topic_;
    std::string camera_frame_;
    std::string lidar_frame_;
    std::string world_frame_;
    std::string car_model_path_;
    std::string armor_model_path_;
    std::vector<int> armor_class_color_map_;
    std::vector<int> armor_class_type_map_;
    int car_class_num_ = 1;
    std::vector<int> car_class_filter_;
    double min_publish_confidence_ = 0.0;
    double car_confidence_threshold_ = 0.25;
    double armor_confidence_threshold_ = 0.50;
    double nms_threshold_ = 0.65;
    bool publish_tentative_tracks_ = false;
    bool camera_fallback_enabled_ = false;
    int camera_fallback_hold_ms_ = 500;
    double camera_fallback_min_confidence_ = 0.45;
    double camera_fallback_ground_z_ = 0.0;
    double camera_fallback_pixel_y_ratio_ = 0.85;
    double camera_fallback_min_x_ = 0.0;
    double camera_fallback_max_x_ = 28.0;
    double camera_fallback_min_y_ = -20.0;
    double camera_fallback_max_y_ = 20.0;
    double camera_fallback_max_range_ = 30.0;
    double tracker_observation_noise_xy_ = 0.18;
    double tracker_observation_noise_z_ = 0.18;
    int tracker_init_thresh_ = 2;
    int tracker_miss_thresh_ = 10;
    double tracker_max_acceleration_ = 5.0;
    double tracker_acceleration_correlation_time_ = 0.6;
    double tracker_distance_weight_ = 0.55;
    double tracker_feature_weight_ = 0.45;
    double tracker_distance_thresh_ = 1.1;
    double tracker_hard_match_distance_ = 2.4;
    double tracker_static_smooth_max_speed_ = 0.12;
    double tracker_static_smooth_radius_ = 0.18;
    double tracker_static_smooth_alpha_ = 0.55;
    bool uav_detection_enabled_ = false;
    bool visualization_enabled_ = true;
    bool visualization_imshow_ = true;
    int visualization_window_width_ = 1280;
    int visualization_window_height_ = 720;
    bool visualization_window_initialized_ = false;
    bool enemy_outpost_detection_enabled_ = false;
    double enemy_outpost_x_ = 17.581;
    double enemy_outpost_y_ = 11.902;
    double enemy_outpost_z_min_ = 5.71;
    double enemy_outpost_z_max_ = 11.10;
    int enemy_outpost_bar_width_ = 24;
    std::string enemy_outpost_alive_topic_;
    int enemy_outpost_min_light_pixels_ = 30;
    double enemy_outpost_min_light_ratio_ = 0.001;
    int enemy_outpost_min_saturation_ = 80;
    int enemy_outpost_min_value_ = 160;
    bool last_enemy_outpost_alive_ = false;
    bool has_enemy_outpost_alive_ = false;
    int sync_queue_size_ = 4;
    int locator_queue_size_ = 2;
    double locator_min_depth_diff_ = 500.0;
    double locator_max_depth_diff_ = 4000.0;
    double locator_cluster_tolerance_ = 400.0;
    int locator_min_cluster_size_ = 8;
    int locator_max_cluster_size_ = 1000;
    double locator_max_distance_ = 29300.0;
    bool locator_adaptive_cluster_enabled_ = false;
    double locator_near_distance_ = 10000.0;
    double locator_mid_distance_ = 18000.0;
    double locator_near_cluster_tolerance_ = 400.0;
    double locator_mid_cluster_tolerance_ = 550.0;
    double locator_far_cluster_tolerance_ = 750.0;
    int locator_near_min_cluster_size_ = 8;
    int locator_mid_min_cluster_size_ = 6;
    int locator_far_min_cluster_size_ = 4;
    bool locator_debug_enabled_ = false;
    bool sentry_targets_enabled_ = true;
    std::string sentry_targets_topic_;
    int sentry_targets_timeout_ms_ = 500;
    int sentry_targets_max_count_ = 6;
    double sentry_targets_match_dist_ = 0.5;
    bool fixed_slot_guess_enabled_ = false;
    int fixed_slot_guess_switch_interval_ms_ = 500;
    double fixed_slot_guess_radius_m_ = 0.3;
    std::vector<double> fixed_slot_guess_hero_point_ { 12.0, 14.2 };
    std::string fixed_slot_guess_hero_point_side_ = "own";
    std::vector<double> fixed_slot_guess_engineer_point_ { 15.4, 7.3 };
    std::string fixed_slot_guess_engineer_point_side_ = "enemy";
    struct FixedSlotGuessState {
        size_t candidate_index = 0;
        rclcpp::Time last_switch_time {};
    } fixed_slot_guess_state_;
    bool early_all_enemy_guess_enabled_ = false;
    std::vector<double> early_all_enemy_guess_point_ { 10.8, 4.2 };
    std::string early_all_enemy_guess_point_side_ = "enemy";
    double early_all_enemy_guess_radius_m_ = 0.3;
    int early_all_enemy_guess_switch_interval_ms_ = 500;
    int early_all_enemy_guess_min_remain_time_sec_ = 240;
    int early_all_enemy_guess_max_remain_time_sec_ = 420;
    uint16_t latest_remain_time_sec_ = 0;
    bool has_remain_time_ = false;
    struct EarlyAllEnemyGuessState {
        size_t candidate_index = 0;
        rclcpp::Time last_switch_time {};
    } early_all_enemy_guess_state_;
    bool late_all_enemy_guess_enabled_ = false;
    std::vector<double> late_all_enemy_guess_points_ { 12.1, 8.6, 12.0, 9.5, 10.7, 8.0 };
    std::string late_all_enemy_guess_point_side_ = "enemy";
    double late_all_enemy_guess_radius_m_ = 0.2;
    int late_all_enemy_guess_switch_interval_ms_ = 500;
    int late_all_enemy_guess_min_remain_time_sec_ = 0;
    int late_all_enemy_guess_max_remain_time_sec_ = 240;
    struct LateAllEnemyGuessState {
        size_t candidate_index = 0;
        rclcpp::Time last_switch_time {};
    } late_all_enemy_guess_state_;
    radar_interface::team_color::ENUM team_color_ = radar_interface::team_color::C_BLUE;
    mutable uint64_t next_fallback_id_ = 1000000;
    bool locator_tf_ready_ = false;
    Eigen::Matrix4d last_lidar_to_camera_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d last_world_to_camera_ = Eigen::Matrix4d::Identity();

    std::mutex init_mutex_;
    std::mutex sentry_targets_mutex_;
    std::optional<sensor_msgs::msg::CameraInfo> camera_info_;
    radar_interface::msg::TargetArray latest_sentry_targets_;
    rclcpp::Time latest_sentry_targets_time_ {};
    bool has_sentry_targets_ = false;
    std::array<CameraFallbackSlot, kRobotsPerTeam * 2> camera_fallback_slots_;
    std::unique_ptr<CvDnnRobotDetector> detector_;
    std::unique_ptr<radar::Locator> locator_;
    std::unique_ptr<radar::Tracker> tracker_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<radar_interface::msg::TargetArray>::SharedPtr sentry_targets_sub_;
    rclcpp::Subscription<radar_interface::team_color::msg>::SharedPtr team_color_sub_;
    rclcpp::Subscription<std_msgs::msg::UInt16>::SharedPtr remain_time_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    std::shared_ptr<Sync> sync_;
    rclcpp::Publisher<radar_interface::msg::TargetArray>::SharedPtr target_pub_;
    rclcpp::Publisher<radar_interface::msg::DetectedTargetArray>::SharedPtr detected_pub_;
    rclcpp::Publisher<radar_interface::msg::MatchResult>::SharedPtr match_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr uav_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enemy_outpost_alive_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RmRadarPipelineNode>());
    rclcpp::shutdown();
    return 0;
}
