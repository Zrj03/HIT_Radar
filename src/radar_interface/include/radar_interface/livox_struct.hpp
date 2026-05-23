#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <cstring>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace radar_interface {
#pragma pack(push, 1)
typedef struct {
    float x; /**< X axis, Unit:m */
    float y; /**< Y axis, Unit:m */
    float z; /**< Z axis, Unit:m */
    float reflectivity; /**< Reflectivity   */
    uint8_t tag; /**< Livox point tag   */
    uint8_t resv; /**< Reserved   */
    double timestamp; /**< Timestamp of point*/
} LivoxPointXyzrtlt;
#pragma pack(pop)

inline const sensor_msgs::msg::PointField* find_lidar_field(
    const sensor_msgs::msg::PointCloud2& msg, const char* name)
{
    for (const auto& field : msg.fields) {
        if (field.name == name) {
            return &field;
        }
    }
    return nullptr;
}

inline bool check_lidar_msg(const sensor_msgs::msg::PointCloud2& msg)
{
    // Size check
    if (msg.point_step == 0) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud point_step error: %d", msg.point_step);
        return false;
    }
    if (msg.row_step != msg.point_step * msg.width) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud row_step error: %d", msg.row_step);
        return false;
    }
    if (msg.data.size() != msg.row_step * msg.height) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud data size error: %lu", msg.data.size());
        return false;
    }
    // Field check (兼容不同livox点格式: 18-byte/26-byte等)
    auto field_x = find_lidar_field(msg, "x");
    auto field_y = find_lidar_field(msg, "y");
    auto field_z = find_lidar_field(msg, "z");
    if (!field_x || !field_y || !field_z) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud required xyz fields missing");
        return false;
    }
    if (field_x->datatype != sensor_msgs::msg::PointField::FLOAT32 || field_x->count != 1 || field_x->offset + sizeof(float) > msg.point_step) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud field x format error");
        return false;
    }
    if (field_y->datatype != sensor_msgs::msg::PointField::FLOAT32 || field_y->count != 1 || field_y->offset + sizeof(float) > msg.point_step) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud field y format error");
        return false;
    }
    if (field_z->datatype != sensor_msgs::msg::PointField::FLOAT32 || field_z->count != 1 || field_z->offset + sizeof(float) > msg.point_step) {
        RCLCPP_ERROR(rclcpp::get_logger("check_lidar_msg"), "point cloud field z format error");
        return false;
    }
    return true;
}
}