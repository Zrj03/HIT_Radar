#include <cv_bridge/cv_bridge.h>
#include <detector/detector_node.h>
#include <utils/common.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace nn_detector;

DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : Node("nn_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting Robomaster Detector Node!");

    bool use_intra = options.use_intra_process_comms();
    if (!use_intra) {
        RCLCPP_WARN(get_logger(), "Not In Intra Process Mode");
    }

    detector_params_.armor_config = declare_parameter("armor_detector_config", "");
    detector_params_.enable_imshow = declare_parameter("enable_imshow", false);
    detector_params_.debug = declare_parameter("debug", false);

    detector_params_.node_dir = ament_index_cpp::get_package_share_directory("nn_detector");
    detector_params_.logger = get_logger();

    // 注册装甲板检测服务
    detect_service = this->create_service<radar_interface::srv::Detect>(
        "detect_armor", std::bind(&DetectorNode::detect_service_callback, this,
                                  std::placeholders::_1, std::placeholders::_2));

    // TF 和 UAV 检测器初始化
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    uav_detector_ = std::make_shared<UAVDetector>(this->get_logger());

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "lidar_mid70/livox/pointcloud", 10,
        std::bind(&DetectorNode::point_cloud_callback, this, std::placeholders::_1));

    radar_warn_pub_ = this->create_publisher<radar_interface::msg::RadarWarn>("/lidar_detect", 10);
    uav_pos_pub_    = this->create_publisher<geometry_msgs::msg::PointStamped>("/radar/uav_target", 10);
    debug_other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/livox/lidar_other", 10);
}

void DetectorNode::ensure_detector_ready()
{
    if (!core) {
        core = std::make_shared<DetectorLib>(detector_params_);
    }
}

void DetectorNode::point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    radar_interface::msg::RadarWarn warn_msg;
    pcl::PointCloud<pcl::PointXYZ> other_cloud;
    geometry_msgs::msg::Point uav_pos;

    uav_detector_->detect(msg, tf_buffer_, warn_msg, other_cloud, uav_pos);

    radar_warn_pub_->publish(warn_msg);

    // 当检测到目标时发布位置
    if (warn_msg.fly_state > 0 || warn_msg.dart_state > 0) {
        geometry_msgs::msg::PointStamped p_msg;
        p_msg.header = msg->header;
        p_msg.point = uav_pos;
        uav_pos_pub_->publish(p_msg);
    }

    // 发布调试点云
    if (debug_other_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(other_cloud, output);
        output.header.frame_id = "rm_frame";
        output.header.stamp = msg->header.stamp;
        debug_other_pub_->publish(output);
    }
}

void DetectorNode::detect_service_callback(const radar_interface::srv::Detect::Request::SharedPtr req,
                                           radar_interface::srv::Detect::Response::SharedPtr rep)
{
    if (req->image.height == 0 || req->image.width == 0 || req->image.data.empty()) {
        RCLCPP_WARN(this->get_logger(), "detect_armor request has empty image, skip inference.");
        return;
    }

    ensure_detector_ready();

    cv::Mat img(req->image.height, req->image.width, encoding2mat_type(req->image.encoding),
                req->image.data.data());
    *rep = *core->detect(img);
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nn_detector::DetectorNode)
