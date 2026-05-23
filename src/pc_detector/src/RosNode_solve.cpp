#include "RosNode.h"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/qos.hpp>
#include <unordered_set>


using namespace pc_detector;

void DetectorNode::update_parameters()
{
    KalmanFilter::read_params(
        get_parameter("kalman_filter.q.position").as_double(),
        get_parameter("kalman_filter.q.velocity").as_double(),
        get_parameter("kalman_filter.q.pos_vel").as_double(),
        get_parameter("kalman_filter.r.position").as_double(),
        get_parameter("kalman_filter.r.velocity").as_double(),
        get_parameter("kalman_filter.decay_rate").as_double(),
        get_parameter("kalman_filter.max_velocity").as_double(),
        get_parameter("kalman_filter.cov_factor").as_double(),
        get_parameter("kalman_filter.stop_p_time").as_int());
    target_map.set_params({
        reinterpret_cast<rclcpp::Node*>(this),
        static_cast<size_t>(get_parameter("target_map.last_frames").as_int()),
        get_parameter("target_map.dist_thres").as_double(),
        static_cast<size_t>(get_parameter("target_map.combine_limit").as_int()),
        get_parameter("target_map.combine_dist").as_double(),
        get_parameter("target_map.force_combine_dist").as_double(),
        static_cast<size_t>(get_parameter("target_map.separate_limit").as_int()),
        get_parameter("target_map.cc_thres").as_double(),
        static_cast<size_t>(get_parameter("target_map.init_lost").as_int()),
        static_cast<size_t>(get_parameter("target_map.min_new_target_points").as_int()),
        get_parameter("target_map.min_new_target_size_x").as_double(),
        get_parameter("target_map.min_new_target_size_y").as_double(),
        get_parameter("target_map.min_new_target_area").as_double(),
        static_cast<size_t>(get_parameter("target_map.min_new_target_confirmations").as_int()),
        static_cast<size_t>(get_parameter("target_map.new_target_confirm_max_gap").as_int()),
        get_parameter("target_map.new_target_candidate_dist").as_double(),
        get_parameter("target_map.trail_filter_enabled").as_int() != 0,
        get_parameter("target_map.trail_filter_min_speed").as_double(),
        get_parameter("target_map.trail_filter_back_dist").as_double(),
        get_parameter("target_map.trail_filter_side_dist").as_double(),
        get_parameter("target_map.trail_filter_point_ratio").as_double(),
        get_parameter("target_map.static_smooth_enabled").as_int() != 0,
        get_parameter("target_map.static_smooth_max_speed").as_double(),
        get_parameter("target_map.static_smooth_radius").as_double(),
        get_parameter("target_map.static_smooth_alpha").as_double(),
        get_parameter("target_map.static_smooth_velocity_decay").as_double(),
        get_parameter("clustering.z_zip").as_double(),
        get_parameter("clustering.loose.box_expand").as_double(),
        { get_parameter("clustering.normal.eps").as_double(),
            static_cast<size_t>(get_parameter("clustering.normal.min_points").as_int()) },
        { get_parameter("clustering.loose.eps").as_double(),
            static_cast<size_t>(get_parameter("clustering.loose.min_points").as_int()) },
        { get_parameter("clustering.strict.eps").as_double(),
            static_cast<size_t>(get_parameter("clustering.strict.min_points").as_int()) }});
}

void DetectorNode::pub_solved_pc(const std::vector<Eigen::Vector3d>& points, const std::vector<int>& clustered_labels, const std::vector<int>& tracking_ids)
{
    sensor_msgs::msg::PointCloud2 cloud;
    sensor_msgs::PointCloud2Modifier(cloud).setPointCloud2Fields(
        5,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "clustered_label", 1, sensor_msgs::msg::PointField::INT32,
        "tracking_id", 1, sensor_msgs::msg::PointField::INT32);
    cloud.header.frame_id.assign("world");
    cloud.header.stamp = pc_buffer.back().timestamp;
    cloud.height = 1;
    cloud.width = points.size();
    cloud.row_step = cloud.width * cloud.point_step;
    cloud.is_bigendian = false;
    cloud.is_dense = true;
    cloud.data.resize(cloud.row_step);
    auto iter_x = sensor_msgs::PointCloud2Iterator<float>(cloud, "x");
    auto iter_y = sensor_msgs::PointCloud2Iterator<float>(cloud, "y");
    auto iter_z = sensor_msgs::PointCloud2Iterator<float>(cloud, "z");
    auto iter_cl = sensor_msgs::PointCloud2Iterator<int>(cloud, "clustered_label");
    auto iter_tr = sensor_msgs::PointCloud2Iterator<int>(cloud, "tracking_id");
    for (size_t i = 0; i < points.size(); ++i) {
        *iter_x = points[i](0);
        *iter_y = points[i](1);
        *iter_z = points[i](2);
        *iter_cl = clustered_labels[i];
        *iter_tr = tracking_ids[i];
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_cl;
        ++iter_tr;
    }
    filtered_pc_publisher->publish(cloud);
}

/// @brief 发布聚类质心标注点
void DetectorNode::pub_cluster_centroids(const std::vector<Eigen::Vector3d>& cluster_centroids)
{
    if (cluster_centroids.empty()) {
        return;
    }
    
    sensor_msgs::msg::PointCloud2 centroid_cloud;
    sensor_msgs::PointCloud2Modifier(centroid_cloud).setPointCloud2Fields(
        3,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32);
    
    centroid_cloud.header.frame_id.assign("world");
    centroid_cloud.header.stamp = pc_buffer.back().timestamp;
    centroid_cloud.height = 1;
    centroid_cloud.width = cluster_centroids.size();
    centroid_cloud.row_step = centroid_cloud.width * centroid_cloud.point_step;
    centroid_cloud.is_bigendian = false;
    centroid_cloud.is_dense = true;
    centroid_cloud.data.resize(centroid_cloud.row_step);
    
    auto iter_x = sensor_msgs::PointCloud2Iterator<float>(centroid_cloud, "x");
    auto iter_y = sensor_msgs::PointCloud2Iterator<float>(centroid_cloud, "y");
    auto iter_z = sensor_msgs::PointCloud2Iterator<float>(centroid_cloud, "z");
    
    for (size_t i = 0; i < cluster_centroids.size(); ++i) {
        *iter_x = cluster_centroids[i](0);
        *iter_y = cluster_centroids[i](1);
        *iter_z = cluster_centroids[i](2);
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }
    
    // 创建标记发行器（如果还没有的话）
    if (!centroid_publisher) {
        centroid_publisher = create_publisher<sensor_msgs::msg::PointCloud2>("detected_centroids", rclcpp::SystemDefaultsQoS());
    }
    centroid_publisher->publish(centroid_cloud);
}

bool DetectorNode::is_static_prior_point(const Eigen::Vector3d& pt) const
{
    if (!static_map_prior_enable || !static_map_prior_pc || !static_map_prior_kdtree)
        return false;

    std::vector<int> nn_indices;
    std::vector<double> nn_dist2;
    if (static_map_prior_kdtree->SearchKNN(pt, 1, nn_indices, nn_dist2) <= 0)
        return false;
    return !nn_dist2.empty() && nn_dist2[0] <= static_map_prior_max_nn_dist2;
}

void DetectorNode::pc_recv_callback(const sensor_msgs::msg::PointCloud2& msg, const LidarContext::SharedPtr l_ctx)
{
    if (!radar_interface::check_lidar_msg(msg))
        return;

    try {
        const auto tf_msg = tf_buffer->lookupTransform("world", l_ctx->tf_frame, tf2::TimePointZero, tf2::durationFromSec(0.01));
        const auto new_trans = tf2::transformToEigen(tf_msg);
        const double trans_delta = (new_trans.translation() - l_ctx->trans.translation()).norm();
        const double rot_delta = Eigen::Quaterniond(new_trans.rotation()).angularDistance(Eigen::Quaterniond(l_ctx->trans.rotation()));
        if (trans_delta > 1e-3 || rot_delta > 1e-4) {
            l_ctx->trans = new_trans;
            prepare_voxel_grid(*l_ctx);
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "Updated TF for %s (d_trans=%.4f, d_rot=%.4f)", l_ctx->tf_frame.c_str(), trans_delta, rot_delta);
        }
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Transform refresh failed for %s: %s", l_ctx->tf_frame.c_str(), e.what());
    }

    UnpackedPcMsg unpacked;
    unpacked.timestamp = msg.header.stamp;
    unpacked.points.reserve(msg.height * msg.width);
    if (msg.height * msg.width * msg.point_step != msg.data.size()) {
        RCLCPP_ERROR(get_logger(), "point cloud size error: %lu", msg.data.size());
        return;
    }

    const auto* field_x = radar_interface::find_lidar_field(msg, "x");
    const auto* field_y = radar_interface::find_lidar_field(msg, "y");
    const auto* field_z = radar_interface::find_lidar_field(msg, "z");
    if (!field_x || !field_y || !field_z)
        return;

    const uint8_t* raw = msg.data.data();
    for (size_t i = 0; i < msg.height * msg.width; ++i)
    {
        const uint8_t* point = raw + i * msg.point_step;
        float x, y, z;
        std::memcpy(&x, point + field_x->offset, sizeof(float));
        std::memcpy(&y, point + field_y->offset, sizeof(float));
        std::memcpy(&z, point + field_z->offset, sizeof(float));
        Eigen::Vector3d pt(x, y, z);
        pt = l_ctx->trans * pt;
        // RCLCPP_INFO(get_logger(), "pt: %f, %f, %f, tag=%d, occc=%d", pt(0), pt(1), pt(2), points[i].tag, voxel_grid.is_occupied(pt));
        // 过滤 https://www.livoxtech.com/cn/showcase/livox-tag
        // if (!voxel_grid->is_occupied(pt) && (points[i].tag & 0b111111) == 0)
        if (!l_ctx->voxel_grid.is_occupied(pt) && !is_static_prior_point(pt))
            unpacked.points.emplace_back(std::move(pt));
    }
    // pub_filtered_pc(unpacked.points);
    pc_buffer.emplace_back(std::move(unpacked));

    while (pc_buffer.size() > static_cast<size_t>(get_parameter("node.buffer_frame_size").as_int()))
        pc_buffer.pop_front();

    update_parameters();
    solve();    // 一收到就解算, 通过发送频率控制
}

void DetectorNode::pub_targets(const rclcpp::Time& time)
{
    visualization_msgs::msg::MarkerArray marker_array;
    radar_interface::msg::TargetArray target_array;
    target_array.header.frame_id.assign("world");
    target_array.header.stamp = time;
    marker_array.markers.reserve(target_map.target_map.size() * 4);
    for (const auto& [id, target] : target_map.target_map) {
        visualization_msgs::msg::Marker marker_sphere_rel, marker_sphere_kalman;
        visualization_msgs::msg::Marker marker_center_sphere;
        radar_interface::msg::Target target_msg;
        float size = 0.2;
        {
            marker_sphere_rel.header.frame_id.assign("world");
            marker_sphere_rel.header.stamp = time;
            marker_sphere_rel.ns = "rel_pos";
            marker_sphere_rel.id = id * 2 + 1;
            marker_sphere_rel.type = visualization_msgs::msg::Marker::SPHERE;
            marker_sphere_rel.scale.x = size;
            marker_sphere_rel.scale.y = size;
            marker_sphere_rel.scale.z = size;
            marker_sphere_rel.pose.position.x = target.grav(0);
            marker_sphere_rel.pose.position.y = target.grav(1);
            marker_sphere_rel.pose.position.z = target.grav(2);
            marker_sphere_rel.pose.orientation.w = 1.;
            marker_sphere_rel.pose.orientation.x = 0.;
            marker_sphere_rel.pose.orientation.y = 0.;
            marker_sphere_rel.pose.orientation.z = 0.;
            marker_sphere_rel.color.a = 0.8;
            marker_sphere_rel.color.r = 1.0;
            marker_sphere_rel.color.g = 0.0;
            marker_sphere_rel.color.b = 0.0;
            marker_sphere_rel.lifetime = rclcpp::Duration::from_seconds(0.1);
        }{
            marker_sphere_kalman.header.frame_id.assign("world");
            marker_sphere_kalman.header.stamp = time;
            marker_sphere_kalman.ns = "kalman";
            marker_sphere_kalman.id = id * 2 + 2;
            marker_sphere_kalman.type = visualization_msgs::msg::Marker::SPHERE;
            marker_sphere_kalman.scale.x = size;
            marker_sphere_kalman.scale.y = size;
            marker_sphere_kalman.scale.z = size;
            auto pos = target.pos();
            marker_sphere_kalman.pose.position.x = pos(0);
            marker_sphere_kalman.pose.position.y = pos(1);
            marker_sphere_kalman.pose.position.z = pos(2) + get_parameter("target_map.project_z").as_double();
            marker_sphere_kalman.pose.orientation.w = 1.;
            marker_sphere_kalman.pose.orientation.x = 0.;
            marker_sphere_kalman.pose.orientation.y = 0.;
            marker_sphere_kalman.pose.orientation.z = 0.;
            marker_sphere_kalman.color.a = 0.8;
            marker_sphere_kalman.color.r = 0.0;
            marker_sphere_kalman.color.g = 0.0;
            marker_sphere_kalman.color.b = 1.0;
            marker_sphere_kalman.lifetime = rclcpp::Duration::from_seconds(0.1);
        }{
            auto pos = target.pos();
            marker_center_sphere.header.frame_id.assign("world");
            marker_center_sphere.header.stamp = time;
            marker_center_sphere.ns = "cluster_center";
            marker_center_sphere.id = id;
            marker_center_sphere.type = visualization_msgs::msg::Marker::SPHERE;
            marker_center_sphere.scale.x = 0.18;
            marker_center_sphere.scale.y = 0.18;
            marker_center_sphere.scale.z = 0.18;
            marker_center_sphere.pose.position.x = pos(0);
            marker_center_sphere.pose.position.y = pos(1);
            marker_center_sphere.pose.position.z = pos(2) + get_parameter("target_map.project_z").as_double();
            marker_center_sphere.pose.orientation.w = 1.;
            marker_center_sphere.pose.orientation.x = 0.;
            marker_center_sphere.pose.orientation.y = 0.;
            marker_center_sphere.pose.orientation.z = 0.;
            marker_center_sphere.color.a = 0.8;
            marker_center_sphere.color.r = 1.0;
            marker_center_sphere.color.g = 1.0;
            marker_center_sphere.color.b = 0.0;
            marker_center_sphere.lifetime = rclcpp::Duration::from_seconds(0.1);
        }{
            target_msg.id = id;
            // target_msg.position.x = target.grav(0);
            // target_msg.position.y = target.grav(1);
            // target_msg.position.z = target.grav(2);
            // target_msg.velocity.x = target.kf.velocity_rel()(0);
            // target_msg.velocity.y = target.kf.velocity_rel()(1);
            // target_msg.velocity.z = 0;
            target_msg.position[0] = target.kf.X(0);
            target_msg.position[1] = target.kf.X(1);
            target_msg.calc_z = target.pos()(2) + get_parameter("target_map.project_z").as_double();
            target_msg.velocity[0] = target.kf.X(2);
            target_msg.velocity[1] = target.kf.X(3);
            target_msg.pos_covariance[0] = target.kf.P(0, 0);
            target_msg.pos_covariance[1] = target.kf.P(0, 1);
            target_msg.pos_covariance[2] = target.kf.P(1, 0);
            target_msg.pos_covariance[3] = target.kf.P(1, 1);
            target_msg.vel_covariance[0] = target.kf.P(2, 2);
            target_msg.vel_covariance[1] = target.kf.P(2, 3);
            target_msg.vel_covariance[2] = target.kf.P(3, 2);
            target_msg.vel_covariance[3] = target.kf.P(3, 3);
            target_msg.uncertainty = target.lost_time;
        }
        marker_sphere_rel.action = visualization_msgs::msg::Marker::MODIFY; // ADD
        marker_sphere_kalman.action = visualization_msgs::msg::Marker::MODIFY; // ADD
        marker_center_sphere.action = visualization_msgs::msg::Marker::MODIFY; // ADD
        marker_array.markers.push_back(marker_sphere_rel);
        marker_array.markers.push_back(marker_sphere_kalman);
        marker_array.markers.push_back(marker_center_sphere);
        target_array.targets.push_back(target_msg);
    }
    marker_publisher->publish(marker_array);
    target_publisher->publish(target_array);
}

void DetectorNode::solve()
{
    open3d::geometry::PointCloud pc;
    for (const auto& unpacked : pc_buffer)
        pc.points_.insert(pc.points_.end(), unpacked.points.begin(), unpacked.points.end());
    std::vector<int> clustered_labels, tracking_ids;
    std::vector<Eigen::Vector3d> cluster_centroids;
    target_map.update(pc, clustered_labels, tracking_ids, cluster_centroids);
    pub_solved_pc(pc.points_, clustered_labels, tracking_ids);
    pub_cluster_centroids(cluster_centroids);
    pub_targets(pc_buffer.back().timestamp);
}
