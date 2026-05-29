#include "pc_aligner/aligner_node.hpp"
#include <tf2_eigen/tf2_eigen.hpp>

#include <cmath>
#include <limits>
#include <map>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>

namespace
{
struct AngularGridPoint
{
    size_t index = 0;
    double distance_sq = -1.0;
};

std::shared_ptr<open3d::geometry::PointCloud> keep_angular_farthest_points(
    const std::shared_ptr<open3d::geometry::PointCloud>& cloud,
    double grid_deg)
{
    if (!cloud || cloud->points_.empty() || !std::isfinite(grid_deg) || grid_deg <= 0.0)
        return cloud;

    std::map<std::pair<int, int>, AngularGridPoint> grid;
    const double rad_to_deg = 180.0 / M_PI;
    for (size_t i = 0; i < cloud->points_.size(); ++i) {
        const auto& point = cloud->points_[i];
        if (!point.allFinite())
            continue;

        const double xy_norm = std::hypot(point.x(), point.y());
        const double distance_sq = point.squaredNorm();
        if (distance_sq <= std::numeric_limits<double>::epsilon())
            continue;

        const int azimuth_index = static_cast<int>(std::floor(std::atan2(point.y(), point.x()) * rad_to_deg / grid_deg));
        const int elevation_index = static_cast<int>(std::floor(std::atan2(point.z(), xy_norm) * rad_to_deg / grid_deg));
        auto& cell = grid[{azimuth_index, elevation_index}];
        if (distance_sq > cell.distance_sq) {
            cell.index = i;
            cell.distance_sq = distance_sq;
        }
    }

    auto filtered = std::make_shared<open3d::geometry::PointCloud>();
    filtered->points_.reserve(grid.size());
    const bool has_colors = cloud->HasColors();
    if (has_colors)
        filtered->colors_.reserve(grid.size());

    for (const auto& [_, cell] : grid) {
        filtered->points_.push_back(cloud->points_[cell.index]);
        if (has_colors)
            filtered->colors_.push_back(cloud->colors_[cell.index]);
    }
    return filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr to_pcl_cloud(const open3d::geometry::PointCloud& cloud)
{
    auto pcl_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl_cloud->reserve(cloud.points_.size());
    for (const auto& point : cloud.points_) {
        if (point.allFinite())
            pcl_cloud->push_back(pcl::PointXYZ(
                static_cast<float>(point.x()),
                static_cast<float>(point.y()),
                static_cast<float>(point.z())));
    }
    return pcl_cloud;
}

double compute_inlier_fitness(
    const pcl::PointCloud<pcl::PointXYZ>& aligned_source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    double max_corr_dist)
{
    if (aligned_source.empty() || !target || target->empty() || !std::isfinite(max_corr_dist) || max_corr_dist <= 0.0)
        return 0.0;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target);

    const float max_corr_dist_sq = static_cast<float>(max_corr_dist * max_corr_dist);
    size_t inliers = 0;
    std::vector<int> indices(1);
    std::vector<float> distances_sq(1);
    for (const auto& point : aligned_source) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;
        if (kdtree.nearestKSearch(point, 1, indices, distances_sq) > 0 && distances_sq[0] <= max_corr_dist_sq)
            ++inliers;
    }

    return static_cast<double>(inliers) / static_cast<double>(aligned_source.size());
}

void log_bounds(
    const rclcpp::Logger& logger,
    const char* name,
    const std::shared_ptr<open3d::geometry::PointCloud>& cloud)
{
    if (!cloud || cloud->points_.empty())
        return;
    const auto min_bound = cloud->GetMinBound();
    const auto max_bound = cloud->GetMaxBound();
    RCLCPP_DEBUG(logger,
        "%s bounds: min=[%.3f, %.3f, %.3f], max=[%.3f, %.3f, %.3f], points=%zu",
        name,
        min_bound.x(), min_bound.y(), min_bound.z(),
        max_bound.x(), max_bound.y(), max_bound.z(),
        cloud->points_.size());
}

std::shared_ptr<open3d::geometry::PointCloud> flattened_z_copy(
    const std::shared_ptr<open3d::geometry::PointCloud>& cloud,
    double z)
{
    auto flattened = std::make_shared<open3d::geometry::PointCloud>(*cloud);
    for (auto& point : flattened->points_)
        point.z() = z;
    if (flattened->HasNormals()) {
        for (auto& normal : flattened->normals_)
            normal = Eigen::Vector3d::UnitZ();
    }
    return flattened;
}
}

void AlignerNode::auto_align(std::shared_ptr<open3d::geometry::PointCloud> sample_pc, Eigen::Isometry3d middle_trans, bool keep_last)
{
    RCLCPP_INFO(get_logger(), "Auto align...");
    // 将 radar 坐标系转换成 pnp 坐标系
    sample_pc->Transform(middle_trans.matrix());

    if (get_parameter("auto_align.use_angular_farthest").as_bool()) {
        const size_t before_size = sample_pc->points_.size();
        sample_pc = keep_angular_farthest_points(
            sample_pc,
            get_parameter("auto_align.angular_grid_deg").as_double());
        RCLCPP_DEBUG(get_logger(), "Angular farthest preprocessing: %zu -> %zu points",
            before_size, sample_pc ? sample_pc->points_.size() : 0);
    }

    std::shared_ptr<open3d::geometry::PointCloud> to_align_pc;

    if (align_using_mesh) {
        // 将 mesh 变成点云，传入参数为点云中点的数量
        auto mesh_pc = mesh_ori->SamplePointsUniformly(get_parameter("mesh_sample").as_int());
        to_align_pc = std::make_shared<open3d::geometry::PointCloud>();
        for (size_t i = 0; i < mesh_pc->points_.size(); ++i) {
            // TODO: 需要修改 z 轴余量大小
            // 将 mesh 中 z>0 部分截取出来
            if (mesh_pc->points_.at(i).z() >= -1e-3) {
                to_align_pc->points_.push_back(mesh_pc->points_.at(i));
                to_align_pc->normals_.push_back(mesh_pc->normals_.at(i));
            }
        }
        // open3d::visualization::DrawGeometries({cropped_mesh_pc}, "cropped_mesh_pc", 1920, 1080);
    } else {
        to_align_pc = pc_align;
    }
    if (!to_align_pc || to_align_pc->points_.empty()) {
        RCLCPP_ERROR(get_logger(), "Alignment model pointcloud is empty, abort auto align.");
        return;
    }
    to_align_pc->PaintUniformColor({1., 0.3, 0.3});
    auto min = get_parameter("crop_box.min").as_double_array();
    auto max = get_parameter("crop_box.max").as_double_array();

    open3d::geometry::AxisAlignedBoundingBox aabb = {
        { min[0], min[1], min[2] },
        { max[0], max[1], max[2] },
    };
    auto cropped = sample_pc->Crop(aabb);
    const double voxel_size = get_parameter("auto_align.voxel_size").as_double();
    if (std::isfinite(voxel_size) && voxel_size > 0.0) {
        const size_t before_size = cropped->points_.size();
        cropped = cropped->VoxelDownSample(voxel_size);
        RCLCPP_DEBUG(get_logger(), "Auto align source voxel downsample: %zu -> %zu points",
            before_size, cropped->points_.size());
    }
    if (cropped->points_.empty()) {
        RCLCPP_WARN(get_logger(), "Auto align cropped pointcloud is empty, skip this alignment.");
        return;
    }
    log_bounds(get_logger(), "Auto align source", cropped);
    log_bounds(get_logger(), "Auto align target", to_align_pc);

    auto source_for_registration = cropped;
    auto target_for_registration = to_align_pc;
    if (get_parameter("auto_align.flatten_z").as_bool()) {
        source_for_registration = flattened_z_copy(cropped, 0.0);
        target_for_registration = flattened_z_copy(to_align_pc, 0.0);
        RCLCPP_DEBUG(get_logger(), "Auto align flatten_z enabled: registering in XY plane.");
    }

    // open3d::visualization::DrawGeometries({cropped}, "cropped_sample_pc", 1920, 1080);
    // open3d::visualization::DrawGeometries({sample_pc, cropped, mesh_ori}, "cropped_origin_mesh", 1920, 1080);

    static Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    if (!keep_last)
        trans = Eigen::Matrix4d::Identity();
    else
        cropped->Transform(trans);

    auto reg = [&](long max_iteration) {
        // 使用 GICP (Generalized ICP) 替代原来的 ICP
        // GICP 在处理噪声和异常值时性能更好，对于点云配准更鲁棒
        return open3d::pipelines::registration::RegistrationGeneralizedICP(
            *source_for_registration, *target_for_registration,
            get_parameter("max_corr_dist").as_double(),
            Eigen::Matrix4d::Identity(),
            open3d::pipelines::registration::TransformationEstimationForGeneralizedICP(),
            open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, max_iteration)); };

    double final_rmse = std::numeric_limits<double>::infinity();
    double final_fitness = 0.0;
    Eigen::Matrix4d step_trans = Eigen::Matrix4d::Identity();
    const auto backend = get_parameter("auto_align.backend").as_string();
    if (backend == "pcl_gicp") {
        auto source_cloud = to_pcl_cloud(*source_for_registration);
        auto target_cloud = to_pcl_cloud(*target_for_registration);
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(source_cloud);
        gicp.setInputTarget(target_cloud);
        gicp.setMaximumIterations(get_parameter("max_iteration").as_int());
        const double max_corr_dist = get_parameter("max_corr_dist").as_double();
        if (std::isfinite(max_corr_dist) && max_corr_dist > 0.0)
            gicp.setMaxCorrespondenceDistance(max_corr_dist);

        pcl::PointCloud<pcl::PointXYZ> aligned;
        gicp.align(aligned);
        const double pcl_score = gicp.getFitnessScore();
        final_rmse = std::sqrt(std::max(0.0, pcl_score));
        final_fitness = compute_inlier_fitness(aligned, target_cloud, max_corr_dist);
        step_trans = gicp.getFinalTransformation().cast<double>();
        RCLCPP_INFO(get_logger(),
            "PCL GICP converged=%s, score=%f, approx_rmse=%f, fitness=%f",
            gicp.hasConverged() ? "true" : "false",
            pcl_score,
            final_rmse,
            final_fitness);
        cropped->Transform(step_trans);
        trans = step_trans * trans;
    } else if (get_parameter("vis_auto").as_bool())
    {
        auto vis = open3d::visualization::Visualizer();
        vis.CreateVisualizerWindow("Auto Aligner", 1920, 1080, 50, 50);
        vis.AddGeometry(cropped);
        vis.AddGeometry(to_align_pc);

        for (unsigned i = 0; i < get_parameter("max_iteration").as_int() && rclcpp::ok(); ++i) {
            auto reg_p2l = reg(1);
            final_rmse = reg_p2l.inlier_rmse_;
            final_fitness = reg_p2l.fitness_;
            vis.UpdateGeometry(cropped);
            vis.PollEvents();
            vis.UpdateRender();
            RCLCPP_INFO(get_logger(), "Align RMSE: %f, Fitness: %f", reg_p2l.inlier_rmse_, reg_p2l.fitness_);
            source_for_registration->Transform(reg_p2l.transformation_);
            cropped->Transform(reg_p2l.transformation_);
            trans = reg_p2l.transformation_ * trans;
        }
    } else {
        auto reg_p2l = reg(get_parameter("max_iteration").as_int());
        final_rmse = reg_p2l.inlier_rmse_;
        final_fitness = reg_p2l.fitness_;
        RCLCPP_INFO(get_logger(), "Align RMSE: %f, Fitness: %f", reg_p2l.inlier_rmse_, reg_p2l.fitness_);
        cropped->Transform(reg_p2l.transformation_);
        trans = reg_p2l.transformation_ * trans;
    }

    if (get_parameter("quality_gate.enable").as_bool()) {
        const double max_rmse = get_parameter("quality_gate.max_rmse").as_double();
        const double min_fitness = get_parameter("quality_gate.min_fitness").as_double();
        if (final_rmse > max_rmse || final_fitness < min_fitness) {
            RCLCPP_WARN(get_logger(),
                "Alignment rejected by quality gate: rmse=%.6f (limit %.6f), fitness=%.6f (limit %.6f). Keep last transform.",
                final_rmse, max_rmse, final_fitness, min_fitness);
            return;
        }
    }

    // to_align_pc->PaintUniformColor({ 1, 1, 1 });
    // open3d::visualization::DrawGeometries({ cropped, cropped_mesh_pc }, "Aligned", 1920, 1080);

    // 发布 对齐过程 坐标系到最终的 world 坐标系的转换
    world_tf = std::make_shared<geometry_msgs::msg::TransformStamped>();
    world_tf->header.stamp = now();
    world_tf->header.frame_id = "middle";
    world_tf->child_frame_id = "world";
    world_tf->transform = tf2::eigenToTransform(Eigen::Affine3d(trans).inverse()).transform;

    tf_broadcaster->sendTransform(*world_tf);
    RCLCPP_INFO(get_logger(), "aligned tf prepared");
}
