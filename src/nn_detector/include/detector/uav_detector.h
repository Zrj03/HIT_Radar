#ifndef _UAV_DETECTOR_H_
#define _UAV_DETECTOR_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <radar_interface/msg/radar_warn.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <thread>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>

namespace nn_detector {

class UAVDetector {
public:
    UAVDetector(const rclcpp::Logger& logger);
    ~UAVDetector() = default;

    void detect(const sensor_msgs::msg::PointCloud2::SharedPtr msg,
                std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                radar_interface::msg::RadarWarn& lidar_detect,
                pcl::PointCloud<pcl::PointXYZ>& other_accumulated_cloud_out,
                geometry_msgs::msg::Point& uav_position);

private:
    rclcpp::Logger logger_;
};

} // namespace nn_detector

#endif
