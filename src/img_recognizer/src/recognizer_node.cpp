#include "img_recognizer/recognizer_node.hpp"
#include "img_recognizer/target_detector.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <sstream>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace img_recognizer;

namespace {

radar_interface::team_color::ENUM parse_team_color(const std::string& color)
{
    if (color == "red" || color == "RED" || color == "Red")
        return radar_interface::team_color::C_RED;
    if (color == "blue" || color == "BLUE" || color == "Blue")
        return radar_interface::team_color::C_BLUE;
    return radar_interface::team_color::UNKNOWN;
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

cv::Rect clamp_square_to_image(const std::pair<cv::Point2d, cv::Point2d>& square, const cv::Mat& image)
{
    const int x1 = std::clamp(static_cast<int>(std::floor(square.first.x)), 0, image.cols);
    const int y1 = std::clamp(static_cast<int>(std::floor(square.first.y)), 0, image.rows);
    const int x2 = std::clamp(static_cast<int>(std::ceil(square.second.x)), 0, image.cols);
    const int y2 = std::clamp(static_cast<int>(std::ceil(square.second.y)), 0, image.rows);
    if (x1 >= x2 || y1 >= y2)
        return {};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

bool has_expected_light(const cv::Mat& image, const std::pair<cv::Point2d, cv::Point2d>& square,
                        radar_interface::team_color::ENUM expected_color,
                        int min_light_pixels, double min_light_ratio,
                        int min_saturation, int min_value)
{
    const cv::Rect roi_rect = clamp_square_to_image(square, image);
    if (roi_rect.empty() || expected_color == radar_interface::team_color::UNKNOWN)
        return false;

    cv::Mat hsv;
    cv::cvtColor(image(roi_rect), hsv, cv::COLOR_BGR2HSV);

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

    const int light_pixels = cv::countNonZero(color_mask);
    const double light_ratio = static_cast<double>(light_pixels) / static_cast<double>(roi_rect.area());
    return light_pixels >= min_light_pixels && light_ratio >= min_light_ratio;
}

} // namespace

RecognizerNode::RecognizerNode(const rclcpp::NodeOptions& options)
    : Node("img_recognizer", options)
{
    declare_parameter("target_topic", "/radar/pc_detector/targets");
    declare_parameter("sync_queue_size", 20);
    declare_parameter("sync_max_interval", 1.0);
    declare_parameter("crop_side_length", 1500.0);
    declare_parameter("crop_center_up_bias_ratio", 0.0);
    declare_parameter("crop_center_right_bias_ratio", 0.0);
    declare_parameter("img_compressed", false);
    declare_parameter("jigsaw_size", 3);
    declare_parameter("img_size", 640);
    declare_parameter("diagnostics_every_n", 30);
    declare_parameter("log_visual_success", true);
    declare_parameter("dump_weak_frames", false);
    declare_parameter("weak_frame_dump_dir", "/tmp/recognizer_weak_frames");
    declare_parameter("weak_frame_dump_every_n", 30);
    declare_parameter("enemy_outpost_detection_enabled", false);
    declare_parameter("enemy_outpost_x", 17.581);
    declare_parameter("enemy_outpost_y", 11.902);
    declare_parameter("enemy_outpost_z", 1.870);
    declare_parameter("enemy_outpost_target_id", 900001);
    declare_parameter("enemy_outpost_alive_topic", "/radar/judge/enemy_outpost_alive");
    declare_parameter("enemy_outpost_min_light_pixels", 30);
    declare_parameter("enemy_outpost_min_light_ratio", 0.001);
    declare_parameter("enemy_outpost_min_saturation", 80);
    declare_parameter("enemy_outpost_min_value", 160);
    declare_parameter("team_color_topic", "/radar/judge/color");
    declare_parameter("default_team_color", "blue");
    declare_parameter("sentry_targets_enabled", true);
    declare_parameter("sentry_targets_topic", "/radar/judge/sentry_targets");
    declare_parameter("sentry_targets_timeout_ms", 500);
    declare_parameter("sentry_targets_default_z", 0.8);

    bool use_intra = options.use_intra_process_comms();
    if (!use_intra) {
        RCLCPP_WARN(get_logger(), "Not In Intra Process Mode");
    }

    auto ns = std::string_view(get_namespace());
    auto ns_pos = ns.rfind('/');
    if (ns_pos != std::string_view::npos && ns_pos + 1 < ns.size()) {
        cam_frame = ns.substr(ns.rfind('/') + 1);
        cam_frame.append("_frame");
    } else {
        cam_frame = "default_camera_frame";
    }

    nn_detector::DetectorParams params;
    // load params
    params.armor_config = declare_parameter("armor_detector_config", "");
    params.enable_imshow = declare_parameter("enable_imshow", false);
    params.debug = declare_parameter("debug", false);

    params.node_dir = ament_index_cpp::get_package_share_directory("nn_detector");
    params.logger = get_logger();
    diagnostics_every_n_ = get_parameter("diagnostics_every_n").as_int();

    tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    detector_lib = std::make_shared<nn_detector::DetectorLib>(params);

    auto ex_name = [this](const std::string& x) {
        return rclcpp::expand_topic_or_service_name(x, get_name(), get_namespace());
    };
    cam_info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::SystemDefaultsQoS(), std::bind(&RecognizerNode::camera_info_callback, this, std::placeholders::_1));
    team_color_sub = this->create_subscription<radar_interface::team_color::msg>(
        get_parameter("team_color_topic").as_string(), rclcpp::SystemDefaultsQoS(),
        std::bind(&RecognizerNode::team_color_callback, this, std::placeholders::_1));
    if (get_parameter("img_compressed").as_bool())
        img_sub.subscribe(this, ex_name("image"), "compressed");
    else
        img_sub.subscribe(this, ex_name("image"), "raw");
    pc_target_sub.subscribe(this, get_parameter("target_topic").as_string());
    sentry_target_sub = this->create_subscription<radar_interface::msg::TargetArray>(
        get_parameter("sentry_targets_topic").as_string(),
        rclcpp::SystemDefaultsQoS(),
        std::bind(&RecognizerNode::sentry_targets_callback, this, std::placeholders::_1));
    sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(get_parameter("sync_queue_size").as_int()), img_sub, pc_target_sub);
    sync->setMaxIntervalDuration(rclcpp::Duration::from_seconds(get_parameter("sync_max_interval").as_double()));
    sync->registerCallback(std::bind(&RecognizerNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

    // markers_pub = this->create_publisher<foxglove_msgs::msg::ImageMarkerArray>("img_recognizer/markers", rclcpp::QoS(rclcpp::KeepLast(10)));
    // annotations_pub = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>("img_recognizer/annotations", rclcpp::QoS(rclcpp::KeepLast(10)));
    detected_targets_pub = this->create_publisher<radar_interface::msg::DetectedTargetArray>("img_recognizer/detected_targets", rclcpp::QoS(rclcpp::KeepLast(10)));
    enemy_outpost_alive_pub = this->create_publisher<std_msgs::msg::Bool>(
        get_parameter("enemy_outpost_alive_topic").as_string(), rclcpp::QoS(rclcpp::KeepLast(10)));

    // nn_helper = std::make_shared<NnHelperNode>(options);

    RCLCPP_INFO(this->get_logger(), "img_recognizer node started.");
}

radar_interface::msg::Target RecognizerNode::make_enemy_outpost_target(const std_msgs::msg::Header&) const
{
    radar_interface::msg::Target target;
    target.id = static_cast<uint64_t>(get_parameter("enemy_outpost_target_id").as_int());
    target.position[0] = get_parameter("enemy_outpost_x").as_double();
    target.position[1] = get_parameter("enemy_outpost_y").as_double();
    target.calc_z = get_parameter("enemy_outpost_z").as_double();
    target.observed_pos[0] = target.position[0];
    target.observed_pos[1] = target.position[1];
    target.observed_pos[2] = target.calc_z;
    return target;
}

void RecognizerNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    // 为什么不用 image_transport::CameraSubscriber? 因为我们已经假定了相机的内参是固定的, 不经常变化
    cam_model.fromCameraInfo(msg);
}

void RecognizerNode::team_color_callback(const radar_interface::team_color::msg::SharedPtr msg)
{
    team_color_ = msg->data ? radar_interface::team_color::C_RED : radar_interface::team_color::C_BLUE;
}

void RecognizerNode::sentry_targets_callback(const radar_interface::msg::TargetArray::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(sentry_targets_mutex_);
    latest_sentry_targets_ = *msg;
    latest_sentry_targets_time_ = now();
    has_sentry_targets_ = true;
}

std::vector<radar_interface::msg::Target> RecognizerNode::get_recent_sentry_targets()
{
    if (!get_parameter("sentry_targets_enabled").as_bool())
        return {};

    std::lock_guard<std::mutex> lock(sentry_targets_mutex_);
    if (!has_sentry_targets_)
        return {};

    const int64_t timeout_ms = get_parameter("sentry_targets_timeout_ms").as_int();
    if (timeout_ms >= 0 &&
        (now() - latest_sentry_targets_time_).nanoseconds() >
            static_cast<int64_t>(timeout_ms) * 1000 * 1000) {
        return {};
    }

    auto targets = latest_sentry_targets_.targets;
    const double default_z = get_parameter("sentry_targets_default_z").as_double();
    for (auto& target : targets) {
        if (target.calc_z == 0.0)
            target.calc_z = default_z;
        target.observed_pos[0] = target.position[0];
        target.observed_pos[1] = target.position[1];
        target.observed_pos[2] = target.calc_z;
    }
    return targets;
}

void RecognizerNode::sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg, const radar_interface::msg::TargetArray::ConstSharedPtr &target_msg)
{
    TargetDetector detector;
    try {
        detector.trans = tf2::transformToEigen(
            tf_buffer->lookupTransform(cam_frame, "world", tf2::TimePointZero, tf2::durationFromSec(1.0)));
    } catch (const tf2::TransformException& e) {
        RCLCPP_ERROR(get_logger(), "%s", e.what());
        return;
    }
    detector.cam_frame = cam_frame;
    detector.node = this->shared_from_this();
    detector.detect_func = std::bind(&nn_detector::DetectorLib::detect, *detector_lib, std::placeholders::_1);
    detector.project_func = [this](const cv::Point3d& cam_pt3) {
        const cv::Point2d uv_rect = cam_model.project3dToPixel(cam_pt3);
        if (!cam_model.initialized()) {
            return uv_rect;
        }
        // 输入图像为 raw 流，投影点需从 rectified 坐标反变换回 raw 坐标。
        return cam_model.unrectifyPoint(uv_rect);
    };
    detector.jigsaw_size = get_parameter("jigsaw_size").as_int();
    detector.img_size = get_parameter("img_size").as_int();

    auto filtered = detector.filter_targets(cv::Point2i(img_msg->width, img_msg->height), target_msg);
    const auto main_filter_stats = detector.last_filter_stats;
    const auto main_rejected_projection_samples = detector.last_rejected_projection_samples;
    const bool enemy_outpost_detection_enabled = get_parameter("enemy_outpost_detection_enabled").as_bool();
    auto detect_targets = filtered;
    const auto sentry_targets = get_recent_sentry_targets();
    if (!sentry_targets.empty()) {
        radar_interface::msg::TargetArray sentry_target_array;
        sentry_target_array.header = target_msg->header;
        sentry_target_array.targets = sentry_targets;
        auto sentry_filtered = detector.filter_targets(cv::Point2i(img_msg->width, img_msg->height), std::make_shared<radar_interface::msg::TargetArray>(sentry_target_array));
        detect_targets.insert(detect_targets.end(), sentry_filtered.begin(), sentry_filtered.end());
        detector.last_filter_stats = main_filter_stats;
        detector.last_rejected_projection_samples = main_rejected_projection_samples;
    }

    auto squares = detector.get_squares(detect_targets);
    const auto main_crop_samples = detector.last_crop_samples;
    auto detect_rep = detector.detect(img_msg, squares);
    auto detected_targets = detector.get_detected_targets(target_msg->header, detect_targets, squares, detect_rep, img_msg->width, img_msg->height);

    if (enemy_outpost_detection_enabled) {
        radar_interface::msg::TargetArray outpost_target_array;
        outpost_target_array.header = target_msg->header;
        outpost_target_array.targets.push_back(make_enemy_outpost_target(target_msg->header));
        auto outpost_filtered = detector.filter_targets(
            cv::Point2i(img_msg->width, img_msg->height),
            std::make_shared<radar_interface::msg::TargetArray>(outpost_target_array));

        bool enemy_outpost_alive = false;
        auto ally_color = team_color_;
        if (ally_color == radar_interface::team_color::UNKNOWN)
            ally_color = parse_team_color(get_parameter("default_team_color").as_string());
        const auto expected_color = enemy_color_for(ally_color);
        if (!outpost_filtered.empty()) {
            const auto outpost_squares = detector.get_squares(outpost_filtered);
            if (!outpost_squares.empty()) {
                const cv::Mat img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
                enemy_outpost_alive = has_expected_light(
                    img, outpost_squares.front(), expected_color,
                    get_parameter("enemy_outpost_min_light_pixels").as_int(),
                    get_parameter("enemy_outpost_min_light_ratio").as_double(),
                    get_parameter("enemy_outpost_min_saturation").as_int(),
                    get_parameter("enemy_outpost_min_value").as_int());
            }
        }
        detector.last_filter_stats = main_filter_stats;
        detector.last_rejected_projection_samples = main_rejected_projection_samples;
        detector.last_crop_samples = main_crop_samples;

        std_msgs::msg::Bool alive_msg;
        alive_msg.data = enemy_outpost_alive;
        enemy_outpost_alive_pub->publish(alive_msg);

        if (!has_enemy_outpost_alive_ || enemy_outpost_alive != last_enemy_outpost_alive_) {
            RCLCPP_INFO(
                get_logger(),
                "Enemy outpost light state: %s expected_color=%s at world=(%.3f,%.3f,%.3f)",
                enemy_outpost_alive ? "alive" : "destroyed",
                color_name(expected_color),
                get_parameter("enemy_outpost_x").as_double(),
                get_parameter("enemy_outpost_y").as_double(),
                get_parameter("enemy_outpost_z").as_double());
            has_enemy_outpost_alive_ = true;
            last_enemy_outpost_alive_ = enemy_outpost_alive;
        }
    }

    int visual_ok = 0;
    int visual_unknown = 0;
    int color_unknown = 0;
    int type_unknown = 0;
    for (const auto& target : detected_targets.targets) {
        const bool known_color = target.color >= 0;
        const bool known_type = target.type >= 0;
        if (known_color && known_type)
            ++visual_ok;
        else
            ++visual_unknown;
        if (!known_color)
            ++color_unknown;
        if (!known_type)
            ++type_unknown;
    }

    if (get_parameter("log_visual_success").as_bool() && visual_ok > 0) {
        for (const auto& detected_target : detected_targets.targets) {
            if (detected_target.color < 0 || detected_target.type < 0)
                continue;
            RCLCPP_INFO(
                get_logger(),
                "Recognizer visual success: target_id=%ld color=%d type=%d pos=(%.2f,%.2f)",
                detected_target.target.id,
                detected_target.color,
                detected_target.type,
                detected_target.target.position[0],
                detected_target.target.position[1]);
        }
    }

    ++sync_callback_count_;
    if (diagnostics_every_n_ > 0 && sync_callback_count_ % diagnostics_every_n_ == 0) {
        RCLCPP_INFO(
            get_logger(),
            "Recognizer summary: pc_targets=%zu filtered_in_image=%zu crops=%zu visual_ok=%d unknown=%d color_unknown=%d type_unknown=%d filter_reject(z<=0:%zu,x:%zu,y:%zu,xy:%zu)",
            target_msg->targets.size(),
            filtered.size(),
            squares.size(),
            visual_ok,
            visual_unknown,
            color_unknown,
            type_unknown,
            detector.last_filter_stats.behind_camera,
            detector.last_filter_stats.x_outside,
            detector.last_filter_stats.y_outside,
            detector.last_filter_stats.xy_outside);

        if (!filtered.empty() && visual_ok == 0) {
            size_t raw_detect_count = 0;
            std::ostringstream raw_detect_stream;
            int sample_count = 0;
            constexpr int MAX_RAW_SAMPLES = 6;
            for (size_t jig_idx = 0; jig_idx < detect_rep.size(); ++jig_idx) {
                const auto& rep = detect_rep[jig_idx];
                raw_detect_count += rep->detected_armors.size();
                for (const auto& armor : rep->detected_armors) {
                    if (sample_count >= MAX_RAW_SAMPLES)
                        break;
                    raw_detect_stream
                        << " [j" << jig_idx
                        << " c=(" << armor.xywh[0] << "," << armor.xywh[1] << ")"
                        << " wh=(" << armor.xywh[2] << "," << armor.xywh[3] << ")"
                        << " conf=" << armor.conf
                        << " color=" << armor.color
                        << " type=" << armor.type
                        << "]";
                    ++sample_count;
                }
                if (sample_count >= MAX_RAW_SAMPLES)
                    break;
            }

            RCLCPP_WARN(
                get_logger(),
                "Recognizer raw detector outputs: jigs=%zu raw_boxes=%zu samples:%s",
                detect_rep.size(),
                raw_detect_count,
                raw_detect_stream.str().c_str());

            std::ostringstream crop_stream;
            for (const auto& sample : detector.last_crop_samples) {
                crop_stream
                    << " [id=" << sample.target_id
                    << " world=(" << sample.world_pt.x() << "," << sample.world_pt.y() << "," << sample.world_pt.z() << ")"
                    << " cam=(" << sample.cam_pt.x() << "," << sample.cam_pt.y() << "," << sample.cam_pt.z() << ")"
                    << " uv=(" << sample.uv.x << "," << sample.uv.y << ")"
                    << " rect=(" << sample.square.first.x << "," << sample.square.first.y
                    << ")-(" << sample.square.second.x << "," << sample.square.second.y << ")"
                    << " side=" << sample.side_length
                    << " dist=" << sample.distance
                    << "]";
            }
            RCLCPP_WARN(
                get_logger(),
                "Recognizer crop samples:%s",
                crop_stream.str().c_str());
        }
    }
    if (target_msg->targets.size() >= 3 && filtered.size() <= 1) {
        std::ostringstream projection_stream;
        for (const auto& sample : detector.last_rejected_projection_samples) {
            projection_stream
                << " [id=" << sample.target_id
                << " reason=" << sample.reason
                << " world=(" << sample.world_pt.x() << "," << sample.world_pt.y() << "," << sample.world_pt.z() << ")"
                << " cam=(" << sample.cam_pt.x() << "," << sample.cam_pt.y() << "," << sample.cam_pt.z() << ")"
                << " uv=(" << sample.uv.x << "," << sample.uv.y << ")]";
        }
        RCLCPP_WARN(
            get_logger(),
            "Recognizer filter bottleneck: pc_targets=%zu filtered_in_image=%zu reject(z<=0:%zu,x:%zu,y:%zu,xy:%zu) samples:%s",
            target_msg->targets.size(),
            filtered.size(),
            detector.last_filter_stats.behind_camera,
            detector.last_filter_stats.x_outside,
            detector.last_filter_stats.y_outside,
            detector.last_filter_stats.xy_outside,
            projection_stream.str().c_str());
    }
    if (filtered.size() >= 3 && visual_ok <= 1) {
        if (get_parameter("dump_weak_frames").as_bool()) {
            const int dump_every_n = get_parameter("weak_frame_dump_every_n").as_int();
            if (dump_every_n > 0 && sync_callback_count_ % dump_every_n == 0) {
                const auto dump_dir = std::filesystem::path(get_parameter("weak_frame_dump_dir").as_string());
                std::error_code ec;
                std::filesystem::create_directories(dump_dir, ec);
                for (size_t i = 0; i < detector.last_jigsaw_images.size(); ++i) {
                    const auto file = dump_dir / (
                        "weak_" + std::to_string(img_msg->header.stamp.sec) + "_" +
                        std::to_string(img_msg->header.stamp.nanosec) + "_" +
                        std::to_string(i) + ".png");
                    cv::imwrite(file.string(), detector.last_jigsaw_images[i]);
                }
                if (ec) {
                    RCLCPP_WARN(get_logger(), "Failed to create weak frame dump dir: %s", ec.message().c_str());
                } else if (!detector.last_jigsaw_images.empty()) {
                    RCLCPP_INFO(
                        get_logger(),
                        "Dumped %zu weak-frame jigsaw images to %s",
                        detector.last_jigsaw_images.size(),
                        dump_dir.c_str());
                }
            }
        }
    }

    detected_targets_pub->publish(detected_targets);
    // markers_pub->publish(detector.get_markers(squares, detect_rep, detected_targets));
    // annotations_pub->publish(detector.get_annotiations(squares, detected_targets));
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(img_recognizer::RecognizerNode)
