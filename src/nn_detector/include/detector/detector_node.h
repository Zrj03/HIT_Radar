#ifndef _DETECTOR_NODE_H
#define _DETECTOR_NODE_H
#include <filesystem>
// ROS
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <radar_interface/srv/detect.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

// detector
#include <detector/detector.h>
#include <detector/detector_lib.h>
#include <detector/net_decoder.h>
#include <detector/uav_detector.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <radar_interface/msg/radar_warn.hpp>

#ifdef TRT
#include <detector/detector_trt.h>
#else
#include <detector/detector_vino.h>
#endif

namespace nn_detector {

class DetectorNode : public rclcpp::Node {
    std::shared_ptr<DetectorLib> core;
    DetectorParams detector_params_;

    rclcpp::Service<radar_interface::srv::Detect>::SharedPtr detect_service;

    void detect_service_callback(const radar_interface::srv::Detect::Request::SharedPtr req,
                                 radar_interface::srv::Detect::Response::SharedPtr rep);

    // UAV Detection
    std::shared_ptr<UAVDetector> uav_detector_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<radar_interface::msg::RadarWarn>::SharedPtr radar_warn_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr uav_pos_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_other_pub_;

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

   public:
    explicit DetectorNode(const rclcpp::NodeOptions& options);
    void ensure_detector_ready();
};
};  // namespace nn_detector

#endif
