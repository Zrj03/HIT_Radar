
#pragma once

/// TargetMap
/// 这是用于储存更新定位目标的类，包括目标的卡尔曼滤波、位置，以及用于可视化的一些东西

#include <cstddef>
#include <deque>
#include <memory>
#include <open3d/Open3D.h>
#include <queue>
#include <vector>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp>

#include "Utility.h"
#include "KalmanFilter.h"
#include "VoxelGrid.h"


namespace pc_detector {
class TargetMap {
public:
    struct Target {
        // size_t id; // 目标 id
        size_t lost_time; // 跟丢计数
        KalmanFilter kf; // 卡尔曼滤波器
        size_t pt_num; // 单次更新中的点数量 (为 0 意味着目标没发生更新)
        Eigen::Vector3d grav; // 单次更新中的重心
        BoundingBox aabb;

        static Z_Map::SharedPtr z_map;
        Eigen::Vector3d pos() const { return z_map->project_ground(kf.pos()); }
    };

    struct TargetMapParams {
        rclcpp::Node* node;
        size_t last_frames;
        double dist_thres;
        size_t combine_limit;
        double combine_dist;
        double force_combine_dist;
        size_t separate_limit;
        double cc_thres;
        size_t init_lost;
        size_t min_new_target_points;
        double min_new_target_size_x;
        double min_new_target_size_y;
        double min_new_target_area;
        size_t min_new_target_confirmations;
        size_t new_target_confirm_max_gap;
        double new_target_candidate_dist;
        bool trail_filter_enabled;
        double trail_filter_min_speed;
        double trail_filter_back_dist;
        double trail_filter_side_dist;
        double trail_filter_point_ratio;
        bool static_smooth_enabled;
        double static_smooth_max_speed;
        double static_smooth_radius;
        double static_smooth_alpha;
        double static_smooth_velocity_decay;
        double z_zip;
        double loose_expand;
        struct {
            double eps;
            size_t min_points;
        } normal, loose, strict;
    };

    struct PendingCandidate {
        BoundingBox aabb;
        size_t pt_num;
        Eigen::Vector3d grav;
        size_t hits;
        size_t last_seen_frame;
    };

private:
    size_t inc_id = 0; // 自增 id
    size_t frame_seq = 0;
    size_t pending_inc_id = 0;
    TargetMapParams params;
    std::unordered_map<size_t, PendingCandidate> pending_candidates;
    std::function<Eigen::Vector3d(const Eigen::Vector3d&)> project_func;
    void element_update(size_t key, const BoundingBox& aabb, size_t pt_num, Eigen::Vector3d grav);
    bool pass_new_target_hard_gate(const BoundingBox& aabb, size_t pt_num) const;
    bool is_trail_cluster(size_t pt_num, const Eigen::Vector3d& grav) const;
    size_t try_new_target_with_confirmation(const BoundingBox& aabb, size_t pt_num, Eigen::Vector3d grav);
    void pre_update();
    void post_update();
    size_t new_target(const BoundingBox& aabb, size_t pt_num, Eigen::Vector3d grav);
    void combine(size_t new_key, std::vector<size_t> old_keys);
    void combine_force(size_t new_key, std::vector<size_t> old_keys);
    void seperate(const BoundingBox& aabb, const open3d::geometry::PointCloud& pc);
    inline bool is_coincide(const BoundingBox& aabb1, const BoundingBox& aabb2)
    {
        /// @brief 判断两个 AABB 是否重合
        // 已二维化
        return std::abs(aabb1.max_bound(0) + aabb1.min_bound(0) - aabb2.max_bound(0) - aabb2.min_bound(0)) <= aabb1.max_bound(0) - aabb1.min_bound(0) + aabb2.max_bound(0) - aabb2.min_bound(0) + params.cc_thres
            && std::abs(aabb1.max_bound(1) + aabb1.min_bound(1) - aabb2.max_bound(1) - aabb2.min_bound(1)) <= aabb1.max_bound(1) - aabb1.min_bound(1) + aabb2.max_bound(1) - aabb2.min_bound(1) + params.cc_thres;
    }
    inline bool is_contain(const BoundingBox& aabb, Eigen::Vector3d pt)
    {
        /// @brief 判断 AABB 是否包含点 pt
        // 已二维化
        return (pt.array() >= aabb.min_bound.array() && pt.array() <= aabb.max_bound.array())(Eigen::seq(0, 1)).all();
    }

public:
    static constexpr int TM_FAILED_INSERT = -1;
    static constexpr int TM_NEED_SEPERATE = -2;
    static constexpr int TM_NOISE = -3;

    void set_params(const TargetMapParams& params);
    void initialize(std::shared_ptr<VoxelGrid> vg);

    static void cluster(const open3d::geometry::PointCloud& pc, double eps_, size_t min_points_, std::vector<int>& labels, std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& pcs,
        open3d::geometry::PointCloud& pc_noise, std::vector<BoundingBox>& aabbs, std::vector<size_t>& pt_num, std::vector<Eigen::Vector3d>& grav);

    static void cluster_d(const open3d::geometry::PointCloud& pc, const Eigen::Vector3d& zero_pos, double eps_, size_t min_points_k,
        std::vector<int>& labels, std::vector<std::shared_ptr<open3d::geometry::PointCloud>>& pcs, open3d::geometry::PointCloud& pc_noise,
        std::vector<BoundingBox>& aabbs, std::vector<size_t>& pt_num, std::vector<Eigen::Vector3d>& grav);

    std::unordered_map<int, Target> target_map;
    std::queue<size_t> discarded_queue;

    size_t push(const BoundingBox& aabb, size_t pt_num, Eigen::Vector3d grav, const open3d::geometry::PointCloud& pc, bool no_strict = false);
    void update(const open3d::geometry::PointCloud& pc, std::vector<int>& cluster_labels, std::vector<int>& tracking_ids, std::vector<Eigen::Vector3d>& cluster_centroids);
    void loose_query(const open3d::geometry::PointCloud& pc);
};
}
