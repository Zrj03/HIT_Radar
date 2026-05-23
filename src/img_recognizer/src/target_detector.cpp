#include <ranges>
#include <cv_bridge/cv_bridge.h>
#include <img_recognizer/target_detector.hpp>
#include <ctime>
#include <cmath>

std::vector<TargetDetector::Target> TargetDetector::filter_targets(const cv::Point2i& img_size, const TargetArray::ConstSharedPtr& msg)
{
    std::vector<TargetDetector::Target> rtn;
    last_filter_stats = {};
    last_rejected_projection_samples.clear();
    last_filter_stats.total = msg->targets.size();
    constexpr double BORDER_MARGIN_RATIO = 0.5;
    const double x_min = -img_size.x * BORDER_MARGIN_RATIO;
    const double y_min = -img_size.y * BORDER_MARGIN_RATIO;
    const double x_max = img_size.x * (1.0 + BORDER_MARGIN_RATIO);
    const double y_max = img_size.y * (1.0 + BORDER_MARGIN_RATIO);
    auto add_projection_sample = [&](const Target& target, const Eigen::Vector3d& world_pt,
                                     const Eigen::Vector3d& cam_pt, const cv::Point2d& uv,
                                     const char* reason) {
        constexpr size_t MAX_PROJECTION_SAMPLES = 4;
        if (last_rejected_projection_samples.size() >= MAX_PROJECTION_SAMPLES)
            return;
        last_rejected_projection_samples.push_back({
            static_cast<int64_t>(target.id),
            world_pt,
            cam_pt,
            uv,
            reason,
        });
    };
    for (auto& target : msg->targets) {
        Eigen::Vector3d world_pt(target.position[0], target.position[1], target.calc_z);
        Eigen::Vector3d cam_pt3_eigen = trans * world_pt;
        if (cam_pt3_eigen.z() <= 0) {
            ++last_filter_stats.behind_camera;
            add_projection_sample(target, world_pt, cam_pt3_eigen, {}, "z<=0");
            continue;
        }
        cv::Point3d cam_pt3(cam_pt3_eigen.x(), cam_pt3_eigen.y(), cam_pt3_eigen.z());
        cv::Point2d cam_pt2 = project_func(cam_pt3);
        const bool x_inside = cam_pt2.x >= x_min && cam_pt2.x < x_max;
        const bool y_inside = cam_pt2.y >= y_min && cam_pt2.y < y_max;
        if (x_inside && y_inside) {
            rtn.emplace_back(target);
            ++last_filter_stats.kept;
            continue;
        }
        if (!x_inside && !y_inside)
            ++last_filter_stats.xy_outside;
        else if (!x_inside)
            ++last_filter_stats.x_outside;
        else
            ++last_filter_stats.y_outside;
        add_projection_sample(target, world_pt, cam_pt3_eigen, cam_pt2,
            (!x_inside && !y_inside) ? "xy" : (!x_inside ? "x" : "y"));
    }
    return rtn;
}

cv::Mat TargetDetector::get_jigsaw_img(const cv::Mat& img, const std::vector<Square>& squares)
{
    cv::Mat jigsaw_img = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    for (unsigned i = 0; i * jigsaw_size < squares.size(); ++i) {
        for (unsigned j = 0; j < jigsaw_size && i * jigsaw_size + j < squares.size(); ++j) {
            auto s = squares[i * jigsaw_size + j];
            if (s.first.x >= s.second.x || s.first.y >= s.second.y)
                continue;
            cv::Mat roi_img;
            if (s.first.x < 0 || s.first.y < 0 || s.second.x >= img.cols || s.second.y >= img.rows) {
                const int square_x = static_cast<int>(std::floor(s.first.x));
                const int square_y = static_cast<int>(std::floor(s.first.y));
                const int square_w = static_cast<int>(std::ceil(s.second.x)) - square_x;
                const int square_h = static_cast<int>(std::ceil(s.second.y)) - square_y;
                if (square_w <= 0 || square_h <= 0)
                    continue;

                const int crop_x1 = std::max(0, square_x);
                const int crop_y1 = std::max(0, square_y);
                const int crop_x2 = std::min(img.cols, square_x + square_w);
                const int crop_y2 = std::min(img.rows, square_y + square_h);
                if (crop_x1 >= crop_x2 || crop_y1 >= crop_y2)
                    continue;

                roi_img = cv::Mat::zeros(square_h, square_w, CV_8UC3);
                cv::Mat roi = img(cv::Rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1));
                const int dst_x = crop_x1 - square_x;
                const int dst_y = crop_y1 - square_y;
                roi.copyTo(roi_img(cv::Rect(dst_x, dst_y, roi.cols, roi.rows)));
            } else {
                cv::Rect roi(s.first.x, s.first.y, s.second.x - s.first.x, s.second.y - s.first.y);
                roi_img = img(roi);
            }
            cv::resize(roi_img, roi_img, cv::Size(img_size / jigsaw_size, img_size / jigsaw_size));
            roi_img.copyTo(jigsaw_img(cv::Rect(j * img_size / jigsaw_size, i * img_size / jigsaw_size, img_size / jigsaw_size, img_size / jigsaw_size)));
        }
    }
    // static cv::VideoWriter writer("detect.mkv", cv::VideoWriter::fourcc('X', '2', '6', '4'), 24, cv::Size(img_size, img_size));
    // writer.write(jigsaw_img);
    // cv::imwrite("dataset/" + std::to_string(std::time(0)) + ".png", jigsaw_img);
    return jigsaw_img;
}

std::vector<TargetDetector::Square> TargetDetector::get_squares(const std::vector<Target>& targets)
{
    double side_length_k = node->get_parameter("crop_side_length").as_double();
    double up_bias_ratio = node->get_parameter("crop_center_up_bias_ratio").as_double();
    double right_bias_ratio = node->get_parameter("crop_center_right_bias_ratio").as_double();
    std::vector<Square> squares;
    last_crop_samples.clear();
    for (const auto& target : targets) {
        Eigen::Vector3d world_pt(target.position[0], target.position[1], target.calc_z);
        Eigen::Vector3d cam_pt3_eigen = trans * world_pt;
        cv::Point3d cam_pt3(cam_pt3_eigen.x(), cam_pt3_eigen.y(), cam_pt3_eigen.z());
        cv::Point2d cam_pt2 = project_func(cam_pt3);
        double distance = std::sqrt(cam_pt3.x * cam_pt3.x + cam_pt3.y * cam_pt3.y + cam_pt3.z * cam_pt3.z);
        double side_length = side_length_k / distance;
        const double center_x = cam_pt2.x + side_length * right_bias_ratio;
        const double center_y = cam_pt2.y - side_length * up_bias_ratio;
        Square square { cv::Point2d(center_x - side_length / 2, center_y - side_length / 2),
            cv::Point2d(center_x + side_length / 2, center_y + side_length / 2) };
        squares.push_back(square);
        constexpr size_t MAX_CROP_SAMPLES = 6;
        if (last_crop_samples.size() < MAX_CROP_SAMPLES) {
            last_crop_samples.push_back({
                static_cast<int64_t>(target.id),
                world_pt,
                cam_pt3_eigen,
                cam_pt2,
                square,
                side_length,
                distance,
            });
        }
    }
    return squares;
}

TargetDetector::DetectRepArray TargetDetector::detect(const Image::ConstSharedPtr& img_msg, const std::vector<Square>& squares)
{
    cv::Mat img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    last_jigsaw_images.clear();
    // 我好想用 C++20 阿啊啊啊啊
    std::vector<Square> one_jigsaw;
    DetectRepArray detect_rep;
    for (const auto& square : squares) {
        if (one_jigsaw.size() == jigsaw_size * jigsaw_size) {
            cv::Mat jigsaw = get_jigsaw_img(img, one_jigsaw);
            last_jigsaw_images.push_back(jigsaw.clone());
            detect_rep.push_back(detect_func(jigsaw));
            one_jigsaw.clear();
        }
        one_jigsaw.push_back(square);
    }
    if (!one_jigsaw.empty()) {
        cv::Mat jigsaw = get_jigsaw_img(img, one_jigsaw);
        last_jigsaw_images.push_back(jigsaw.clone());
        detect_rep.push_back(detect_func(jigsaw));
    }
    return detect_rep;
}

TargetDetector::DetectedTargetArray TargetDetector::get_detected_targets(const std_msgs::msg::Header& header, const std::vector<Target>& targets, const std::vector<Square>& squares, DetectRepArray& detect_rep,
    uint32_t img_width, uint32_t img_height)
{
    // NOTE: 这里会将 armor 的角点转换到相机坐标系下
    assert(targets.size() == squares.size());
    DetectedTargetArray detected_targets;
    detected_targets.header = header;
    // jig_n: 拼图序号
    for (unsigned jig_n = 0; jig_n < detect_rep.size(); ++jig_n) {
        // jig_m: 拼图内序号
        for (unsigned jig_m = 0; jig_n * jigsaw_size * jigsaw_size + jig_m < squares.size() && jig_m < jigsaw_size * jigsaw_size; ++jig_m) {
            radar_interface::msg::DetectedTarget detected_target;
            detected_target.target = targets[jig_n * jigsaw_size * jigsaw_size + jig_m];
            detected_target.color = -1;
            detected_target.type = -1;
            double min_dist_sqr = 1e9;
            unsigned jig_block_size = img_size / jigsaw_size;
            unsigned jig_x = jig_m % jigsaw_size, jig_y = jig_m / jigsaw_size;
            cv::Point2d square_centre((jig_x + 0.5) * jig_block_size, (jig_y + 0.5) * jig_block_size);
            for (auto& target : detect_rep[jig_n]->detected_armors) {
                // 如果目标不在拼图块内, 则跳过
                constexpr double BLOCK_MARGIN_RATIO = 0.2;
                double block_x_min = jig_x * jig_block_size - jig_block_size * BLOCK_MARGIN_RATIO;
                double block_x_max = (jig_x + 1) * jig_block_size + jig_block_size * BLOCK_MARGIN_RATIO;
                double block_y_min = jig_y * jig_block_size - jig_block_size * BLOCK_MARGIN_RATIO;
                double block_y_max = (jig_y + 1) * jig_block_size + jig_block_size * BLOCK_MARGIN_RATIO;
                if (target.xywh[0] < block_x_min || target.xywh[0] > block_x_max ||
                    target.xywh[1] < block_y_min || target.xywh[1] > block_y_max)
                    continue;
                // 转换到以拼图块左上角为原点坐标
                auto to_jig_coord = [&](cv::Point2d pt) {
                    return cv::Point2d(pt.x - jig_x * jig_block_size, pt.y - jig_y * jig_block_size);
                };
                // 缩放, 并位移到真实尺寸
                // FIXME: 在边缘时会飞出框外
                // 好像不是雷达的问题, 是网络的问题
                auto to_real = [&](cv::Point2d pt) {
                    auto now_square = squares[jig_n * jigsaw_size * jigsaw_size + jig_m];
                    return cv::Point2d(pt.x * (double)(now_square.second.x - now_square.first.x) / (double)jig_block_size + now_square.first.x,
                        pt.y * (double)(now_square.second.y - now_square.first.y) / (double)jig_block_size + now_square.first.y);
                };
                for (auto& pt : target.pts) {
                    auto pt_cv = to_real(to_jig_coord({ pt.x, pt.y }));
                    // pt.x = pt_cv.x, pt.y = pt_cv.y;
                    pt.x = std::max(0.0, std::min(pt_cv.x, static_cast<double>(img_width - 1)));
                    pt.y = std::max(0.0, std::min(pt_cv.y, static_cast<double>(img_height - 1)));
                }

                double dist_sqr = (target.xywh[0] - square_centre.x) * (target.xywh[0] - square_centre.x) + (target.xywh[1] - square_centre.y) * (target.xywh[1] - square_centre.y);
                if (dist_sqr < min_dist_sqr) {
                    min_dist_sqr = dist_sqr;
                    detected_target.color = target.color;
                    detected_target.type = (target.type == radar_interface::msg::Armor::TYPE_0)
                        ? -1
                        : target.type;
                }
            }

            if (detected_target.type == -1 || detected_target.color == -1) {
                constexpr double FALLBACK_BLOCK_RADIUS_RATIO = 0.4;
                double fallback_min_dist_sqr = 1e9;
                double fallback_max_dist_sqr = jig_block_size * jig_block_size * FALLBACK_BLOCK_RADIUS_RATIO * FALLBACK_BLOCK_RADIUS_RATIO;
                for (const auto& target : detect_rep[jig_n]->detected_armors) {
                    double dist_sqr = (target.xywh[0] - square_centre.x) * (target.xywh[0] - square_centre.x) + (target.xywh[1] - square_centre.y) * (target.xywh[1] - square_centre.y);
                    if (dist_sqr > fallback_max_dist_sqr)
                        continue;
                    if (dist_sqr < fallback_min_dist_sqr) {
                        fallback_min_dist_sqr = dist_sqr;
                        detected_target.color = target.color;
                        detected_target.type = (target.type == radar_interface::msg::Armor::TYPE_0)
                            ? -1
                            : target.type;
                    }
                }
            }
            detected_targets.targets.push_back(detected_target);
        }
    }
    return detected_targets;
}

#if 0
foxglove_msgs::msg::ImageMarkerArray TargetDetector::get_markers(const std::vector<Square>& squares, const DetectRepArray& detect_rep, const DetectedTargetArray& detected_targets)
{
    assert(squares.size() == detected_targets.targets.size());
    foxglove_msgs::msg::ImageMarkerArray markers;
    // 对象框 (squares)
    for (unsigned i = 0; i < squares.size(); ++i) {
            if (detected_targets.targets[i].type == radar_interface::msg::Armor::TYPE_0 || 
                detected_targets.targets[i].type == radar_interface::msg::Armor::TYPE_SENTRY) {
                continue;
            }
            visualization_msgs::msg::ImageMarker marker;
            marker.header.frame_id = cam_frame;
            marker.header.stamp = detected_targets.header.stamp;
            marker.ns = "squares";
            marker.type = visualization_msgs::msg::ImageMarker::POLYGON;
            marker.action = visualization_msgs::msg::ImageMarker::ADD;
            marker.points.resize(4);
            marker.points[0].x = squares[i].first.x;
            marker.points[0].y = squares[i].first.y;
            marker.points[1].x = squares[i].second.x;
            marker.points[1].y = squares[i].first.y;
            marker.points[2].x = squares[i].second.x;
            marker.points[2].y = squares[i].second.y;
            marker.points[3].x = squares[i].first.x;
            marker.points[3].y = squares[i].second.y;
            marker.outline_color.a = 1.0;
            switch (detected_targets.targets[i].color) {
            case radar_interface::msg::Armor::COLOR_RED:
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 0.0;
                break;
            case radar_interface::msg::Armor::COLOR_BLUE:
                marker.outline_color.r = 0.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 1.0;
                break;
            case radar_interface::msg::Armor::COLOR_PURPLE:
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 1.0;
                break;
            default: // UNKNOWN
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 1.0;
                marker.outline_color.b = 1.0;
                break;
            }
            marker.scale = 3;
            marker.filled = false;
            markers.markers.push_back(marker);
    }
    // 识别结果 (detect_rep)
    for (auto& rep : detect_rep) {
        for (auto& armor : rep->detected_armors) {
            visualization_msgs::msg::ImageMarker marker;
            marker.header.frame_id = cam_frame;
            marker.header.stamp = detected_targets.header.stamp;
            marker.ns = "detect_rep";
            marker.type = visualization_msgs::msg::ImageMarker::POLYGON;
            marker.action = visualization_msgs::msg::ImageMarker::ADD;
            marker.points.resize(4);
            marker.points[0].x = armor.pts[0].x;
            marker.points[0].y = armor.pts[0].y;
            marker.points[1].x = armor.pts[1].x;
            marker.points[1].y = armor.pts[1].y;
            marker.points[2].x = armor.pts[2].x;
            marker.points[2].y = armor.pts[2].y;
            marker.points[3].x = armor.pts[3].x;
            marker.points[3].y = armor.pts[3].y;
            marker.outline_color.a = 1.0;
            switch (armor.color) {
            case radar_interface::msg::Armor::COLOR_RED:
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 0.0;
                break;
            case radar_interface::msg::Armor::COLOR_BLUE:
                marker.outline_color.r = 0.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 1.0;
                break;
            case radar_interface::msg::Armor::COLOR_PURPLE:
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 0.0;
                marker.outline_color.b = 1.0;
                break;
            default: // UNKNOWN
                marker.outline_color.r = 1.0;
                marker.outline_color.g = 1.0;
                marker.outline_color.b = 1.0;
                break;
            }
            marker.scale = 3;
            marker.filled = false;
            markers.markers.push_back(marker);
        }
    }
    return markers;
}
#endif

#if 0
foxglove_msgs::msg::ImageAnnotations TargetDetector::get_annotiations(const std::vector<Square>& squares, const DetectedTargetArray& detected_targets)
{
    assert(squares.size() == detected_targets.targets.size());
    foxglove_msgs::msg::ImageAnnotations annotations;
    annotations.timestamp = detected_targets.header.stamp;
    for (unsigned i = 0; i < squares.size(); ++i) {
        const auto& square = squares[i];
        const auto center_x = (square.first.x + square.second.x) * 0.5;
        const auto center_y = (square.first.y + square.second.y) * 0.5;
        const auto width = std::max(1.0, square.second.x - square.first.x);
        const auto height = std::max(1.0, square.second.y - square.first.y);

        foxglove_msgs::msg::Color outline_color;
        foxglove_msgs::msg::Color fill_color;
        outline_color.a = 1.0;
        fill_color.a = 0.0;
        switch (detected_targets.targets[i].color) {
        case radar_interface::msg::Armor::COLOR_RED:
            outline_color.r = 1.0;
            break;
        case radar_interface::msg::Armor::COLOR_BLUE:
            outline_color.b = 1.0;
            break;
        case radar_interface::msg::Armor::COLOR_PURPLE:
            outline_color.r = 1.0;
            outline_color.b = 1.0;
            break;
        default:
            outline_color.r = 1.0;
            outline_color.g = 1.0;
            outline_color.b = 0.0;
            break;
        }

        foxglove_msgs::msg::PointsAnnotation square_annotation;
        square_annotation.timestamp = detected_targets.header.stamp;
        square_annotation.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;
        square_annotation.thickness = 2.0;
        square_annotation.outline_color = outline_color;
        square_annotation.fill_color = fill_color;
        square_annotation.points.resize(4);
        square_annotation.points[0].x = square.first.x;
        square_annotation.points[0].y = square.first.y;
        square_annotation.points[1].x = square.second.x;
        square_annotation.points[1].y = square.first.y;
        square_annotation.points[2].x = square.second.x;
        square_annotation.points[2].y = square.second.y;
        square_annotation.points[3].x = square.first.x;
        square_annotation.points[3].y = square.second.y;
        annotations.points.push_back(square_annotation);

        foxglove_msgs::msg::CircleAnnotation center_annotation;
        center_annotation.timestamp = detected_targets.header.stamp;
        center_annotation.position.x = center_x;
        center_annotation.position.y = center_y;
        center_annotation.diameter = std::min(width, height) * 0.12;
        center_annotation.thickness = 2.0;
        center_annotation.fill_color = fill_color;
        center_annotation.outline_color = outline_color;
        annotations.circles.push_back(center_annotation);

        if (detected_targets.targets[i].type == radar_interface::msg::Armor::TYPE_0 || 
            detected_targets.targets[i].type == radar_interface::msg::Armor::TYPE_SENTRY) {
            continue;
        }
        
        foxglove_msgs::msg::TextAnnotation annotation;
        annotation.text = "Color: ";
        switch (detected_targets.targets[i].color) {
        case radar_interface::msg::Armor::COLOR_RED:
            annotation.text += "RED";
            annotation.background_color.r = 1.0;
            annotation.background_color.g = 0.0;
            annotation.background_color.b = 0.0;
            annotation.background_color.a = 1.0;
            break;
        case radar_interface::msg::Armor::COLOR_BLUE:
            annotation.text += "BLUE";
            annotation.background_color.r = 0.0;
            annotation.background_color.g = 0.0;
            annotation.background_color.b = 1.0;
            annotation.background_color.a = 1.0;
            break;
        case radar_interface::msg::Armor::COLOR_PURPLE:
            annotation.text += "PURPLE";
            annotation.background_color.r = 1.0;
            annotation.background_color.g = 0.0;
            annotation.background_color.b = 1.0;
            annotation.background_color.a = 1.0;
            break;
        default: // UNKNOWN
            annotation.text += "UNKNOWN";
            annotation.background_color.a = 0.0;
            break;
        }
        annotation.text += " Type: ";
        switch (detected_targets.targets[i].type) {
        case radar_interface::msg::Armor::TYPE_SENTRY:
            annotation.text += "SENTRY";
            break;
        case radar_interface::msg::Armor::TYPE_HERO:
            annotation.text += "HERO";
            break;
        case radar_interface::msg::Armor::TYPE_ENGINEER:
            annotation.text += "ENGINEER";
            break;
        case radar_interface::msg::Armor::TYPE_INF_3:
            annotation.text += "INFANTRY 3";
            break;
        case radar_interface::msg::Armor::TYPE_INF_4:
            annotation.text += "INFANTRY 4";
            break;
        case radar_interface::msg::Armor::TYPE_INF_5:
            annotation.text += "INFANTRY 5";
            break;
        case radar_interface::msg::Armor::TYPE_0:
            annotation.text += "0";
            break;
        case radar_interface::msg::Armor::TYPE_BS:
            annotation.text += "Bs";
            break;
        case radar_interface::msg::Armor::TYPE_BB:
            annotation.text += "Bb";
            break;
        default: // UNKNOWN
            annotation.text += "UNKNOWN";
            break;
        }
        annotation.position.x = square.first.x;
        annotation.position.y = square.first.y - 15;
        annotation.text_color.r = 1.0;
        annotation.text_color.g = 1.0;
        annotation.text_color.b = 1.0;
        annotation.text_color.a = 1.0;
        annotation.font_size = 30;
        annotations.texts.push_back(annotation);
    }
    return annotations;
}
#endif
