#include "pc_aligner/aligner_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <open3d/Open3D.h>
#include <tf2_eigen/tf2_eigen.hpp>

// 选择点云中的点
// @param pcd: 输入点云
// @return 返回用户选择的点的索引
std::vector<size_t> select_points(std::shared_ptr<const open3d::geometry::PointCloud> pcd)
{
    RCLCPP_INFO(rclcpp::get_logger("pick_point"), "Select points...");
    RCLCPP_INFO(rclcpp::get_logger("pick_point"), "  Use [shift + left click] to pick points.");
    RCLCPP_INFO(rclcpp::get_logger("pick_point"), "  Use [shift + right click] to undo point picking.");
    RCLCPP_INFO(rclcpp::get_logger("pick_point"), "  After picking points, press 'Q' to close the window.");

    open3d::visualization::VisualizerWithEditing vis_;
    vis_.CreateVisualizerWindow("Select Points", 1920, 1080);
    vis_.AddGeometry(pcd);
    vis_.Run();
    vis_.DestroyVisualizerWindow();
    return vis_.GetPickedPoints();
}

// 读取预先选择的点云文件
// @return 返回点云对象
std::shared_ptr<const open3d::geometry::PointCloud> AlignerNode::read_preselected_pc()
{
    std::filesystem::path pc_path = std::filesystem::path(ament_index_cpp::get_package_share_directory("radar_bringup")) / "resource" / get_parameter("preselect_pcd").as_string();
    auto pc2align = open3d::io::CreatePointCloudFromFile(pc_path.string());
    return pc2align;
}

// 利用预选择的点进行手动配准获取初始变换矩阵
// @param pc2align: 待配准的点云
// @return 初始变换矩阵
Eigen::Matrix4d AlignerNode::manual_trans_pre(std::shared_ptr<const open3d::geometry::PointCloud> pc2align)
{
    auto preselected = read_preselected_pc();
    pub_add_points(preselected);
    auto picked_pc = select_points(pc2align);
    pub_del_points(preselected->points_.size());
    if (picked_pc.size() < 3)
        throw std::runtime_error("Too few points selected.");
    if (preselected->points_.size() < 3)
        throw std::runtime_error("Too few preselected points.");
    if (picked_pc.size() != preselected->points_.size())
        throw std::runtime_error("Selected points number not match.");
    std::vector<Eigen::Vector2i> correspondences;
    for (size_t i = 0; i < picked_pc.size(); ++i)
        correspondences.emplace_back(picked_pc[i], i);
    open3d::pipelines::registration::TransformationEstimationPointToPoint pointToPoint;
    return pointToPoint.ComputeTransformation(*pc2align, *preselected, correspondences);
}

// 完全手动配准获取初始变换矩阵
// @param pc2align: 待配准的点云
// @param mesh_pc: 模型点云
// @return 初始变换矩阵
Eigen::Matrix4d AlignerNode::manual_trans_mesh(std::shared_ptr<const open3d::geometry::PointCloud> pc2align, std::shared_ptr<const open3d::geometry::PointCloud> mesh_pc)
{
    RCLCPP_INFO(rclcpp::get_logger("manual_align"), "Step 1/2: select points on model point cloud, then press Q.");
    auto picked_mesh = select_points(mesh_pc);
    RCLCPP_INFO(rclcpp::get_logger("manual_align"), "Step 2/2: select corresponding points on lidar point cloud, then press Q.");
    auto picked_pc = select_points(pc2align);
    if (picked_pc.size() < 3 || picked_mesh.size() < 3)
        throw std::runtime_error("Too few points selected.");
    if (picked_pc.size() != picked_mesh.size())
        throw std::runtime_error("Selected points number not match.");
    std::vector<Eigen::Vector2i> correspondences;
    for (size_t i = 0; i < picked_pc.size(); ++i)
        correspondences.emplace_back(picked_pc[i], picked_mesh[i]);
    open3d::pipelines::registration::TransformationEstimationPointToPoint pointToPoint;
    return pointToPoint.ComputeTransformation(*pc2align, *mesh_pc, correspondences);
}

// 手动配准主函数
// @param pc2align_: 输入点云
void AlignerNode::manual_align(std::shared_ptr<open3d::geometry::PointCloud> pc2align_)
{
    RCLCPP_INFO(get_logger(), "Manual align...");

    auto min = get_parameter("manual_crop.min").as_double_array();
    auto max = get_parameter("manual_crop.max").as_double_array();

    open3d::geometry::AxisAlignedBoundingBox aabb = {
        { min[0], min[1], min[2] },
        { max[0], max[1], max[2] },
    };
    auto pc2align = pc2align_->Crop(aabb);
    // pc2align->EstimateNormals();

    Eigen::Matrix4d trans;

    const bool auto_retry = get_parameter("manual_align.auto_retry_on_fail").as_bool();
    const int max_retries = get_parameter("manual_align.max_retries").as_int();
    int retry_count = 0;

    while (rclcpp::ok()) {
        try {
            if (get_parameter("use_preselect").as_bool())
                trans = manual_trans_pre(pc2align);
            else {
                std::shared_ptr<open3d::geometry::PointCloud> model_pc;
                if (align_using_mesh) {
                    model_pc = mesh_ori->SamplePointsUniformly(get_parameter("mesh_sample").as_int());
                    model_pc->colors_.resize(model_pc->points_.size());
                    for (size_t i = 0; i < model_pc->points_.size(); ++i) {
                        if (model_pc->points_[i](0) < 14)
                            model_pc->colors_[i] = { 1., 0., 0. };   // red
                        else
                            model_pc->colors_[i] = { 0., 0., 1. };   // blue
                    }
                } else {
                    model_pc = pc_align;
                }
                trans = manual_trans_mesh(pc2align, model_pc);
            }
            break;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Manual align failed: %s", e.what());
            if (!auto_retry || retry_count >= max_retries) {
                RCLCPP_WARN(get_logger(), "Manual align aborted. Re-call service or restart if you want to retry.");
                return;
            }
            ++retry_count;
            RCLCPP_WARN(get_logger(), "Retrying manual align (%d/%d)...", retry_count, max_retries);
        }
    }

    // 使用 GICP 对手动配准得到的初始变换进行精细化
    RCLCPP_INFO(get_logger(), "Refining manual align with GICP...");
    auto pc2align_copy = std::make_shared<open3d::geometry::PointCloud>(*pc2align);
    pc2align_copy->Transform(trans);

    std::shared_ptr<open3d::geometry::PointCloud> to_align_pc_manual;
    if (align_using_mesh) {
        auto mesh_pc = mesh_ori->SamplePointsUniformly(get_parameter("mesh_sample").as_int());
        to_align_pc_manual = std::make_shared<open3d::geometry::PointCloud>();
        for (size_t i = 0; i < mesh_pc->points_.size(); ++i) {
            if (mesh_pc->points_.at(i).z() >= -1e-3) {
                to_align_pc_manual->points_.push_back(mesh_pc->points_.at(i));
                to_align_pc_manual->normals_.push_back(mesh_pc->normals_.at(i));
            }
        }
    } else {
        to_align_pc_manual = pc_align;
    }

    // 进行 GICP 精细化（较少迭代次数，因为已有初始对齐）
    auto reg_gicp = open3d::pipelines::registration::RegistrationGeneralizedICP(
        *pc2align_copy, *to_align_pc_manual,
        get_parameter("max_corr_dist").as_double(),
        Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationForGeneralizedICP(),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, 5));
    
    trans = reg_gicp.transformation_ * trans;
    RCLCPP_INFO(get_logger(), "Manual GICP refinement - RMSE: %f, Fitness: %f", 
                reg_gicp.inlier_rmse_, reg_gicp.fitness_);

    middle_tf = std::make_shared<geometry_msgs::msg::TransformStamped>();
    middle_tf->header.stamp = now();
    middle_tf->header.frame_id = get_parameter("sample_lidar").as_string() + "_frame";
    middle_tf->child_frame_id = "middle";
    middle_tf->transform = tf2::eigenToTransform(Eigen::Affine3d(trans).inverse()).transform;

    tf_broadcaster->sendTransform(*middle_tf);
    RCLCPP_INFO(get_logger(), "Manual align done (with GICP refinement).");
}

// 发布点云中的点作为可视化标记
// @param pc: 输入点云
void AlignerNode::pub_add_points(std::shared_ptr<const open3d::geometry::PointCloud> pc)
{
    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time time = rclcpp::Clock().now();

    auto add_point = [&](unsigned id, Eigen::Vector3d pos) {
        visualization_msgs::msg::Marker marker_sphere, marker_text;
        float size = 0.5;

        marker_sphere.header.frame_id.assign("world");
        marker_sphere.header.stamp = time;
        marker_sphere.ns = "sphere";
        marker_sphere.id = id;
        marker_sphere.type = visualization_msgs::msg::Marker::SPHERE;
        marker_sphere.action = visualization_msgs::msg::Marker::ADD;
        marker_sphere.scale.x = size;
        marker_sphere.scale.y = size;
        marker_sphere.scale.z = size;
        marker_sphere.pose.position.x = pos(0);
        marker_sphere.pose.position.y = pos(1);
        marker_sphere.pose.position.z = pos(2);
        marker_sphere.pose.orientation.w = 1.;
        marker_sphere.pose.orientation.x = 0.;
        marker_sphere.pose.orientation.y = 0.;
        marker_sphere.pose.orientation.z = 0.;
        if (pos(0) < 14)
            marker_sphere.color.r = 1.0, marker_sphere.color.b = 0.0;
        else
            marker_sphere.color.r = 0.0, marker_sphere.color.b = 1.0;
        marker_sphere.color.a = 0.8;
        marker_sphere.color.g = 0.0;
        marker_array.markers.push_back(marker_sphere);

        marker_text.header.frame_id.assign("world");
        marker_text.header.stamp = time;
        marker_text.ns = "text";
        marker_text.id = id;
        marker_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker_text.action = visualization_msgs::msg::Marker::ADD;
        marker_text.scale.z = 0.8;
        marker_text.pose.position.x = pos(0);
        marker_text.pose.position.y = pos(1);
        marker_text.pose.position.z = pos(2) + size;
        marker_text.pose.orientation.w = 1.;
        marker_text.pose.orientation.x = 0.;
        marker_text.pose.orientation.y = 0.;
        marker_text.pose.orientation.z = 0.;
        marker_text.color.a = 0.8;
        marker_text.color.r = 1.0;
        marker_text.color.g = 1.0;
        marker_text.color.b = 1.0;
        marker_text.text = "point: " + std::to_string(id);
        marker_array.markers.push_back(marker_text);
    };

    for (unsigned i = 0; i < pc->points_.size(); ++i)
        add_point(i, pc->points_[i]);

    marker_pub->publish(marker_array);
}

// 删除点云中的可视化标记
// @param size: 点的数量
void AlignerNode::pub_del_points(unsigned size)
{
    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time time = rclcpp::Clock().now();

    auto remove_point = [&](unsigned id) {
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id.assign("world");
        marker.header.stamp = time;
        marker.ns = "sphere";
        marker.id = id;
        marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(marker);

        marker.ns = "text";
        marker.id = id;
        marker.action = visualization_msgs::msg::Marker::DELETE;
        marker_array.markers.push_back(marker);
    };

    for (unsigned i = 0; i < size; ++i)
        remove_point(i);

    marker_pub->publish(marker_array);
}
