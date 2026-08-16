#include "target_matcher/matcher_node.hpp"
#include "target_matcher/utils.hpp"
#include "target_matcher/visualization.hpp"

#include <cv_bridge/cv_bridge.h>
#include <dlib/optimization.h>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
inline bool is_infantry_idx(int idx)
{
    return (idx >= 3 && idx <= 5) || (idx >= 9 && idx <= 11);
}

inline std::pair<int, int> infantry_idx_range(int idx)
{
    if (idx >= 3 && idx <= 5)
        return { 3, 5 };
    if (idx >= 9 && idx <= 11)
        return { 9, 11 };
    return { -1, -1 };
}

inline std::pair<int, int> dominant_infantry_idx(const std::array<long, 12>& value, int begin, int end)
{
    int best_idx = begin;
    int second_idx = begin;
    for (int idx = begin + 1; idx <= end; ++idx) {
        if (value[idx] > value[best_idx]) {
            second_idx = best_idx;
            best_idx = idx;
        } else if (second_idx == best_idx || value[idx] > value[second_idx]) {
            second_idx = idx;
        }
    }
    return { best_idx, second_idx };
}

}

/// 这里不做消息的时间戳同步，是为了提高实时性

MatcherNode::MatcherNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("matcher", options)
{
    for (unsigned i = 0; i < 6; ++i)
        result.red[i].id = -1, result.blue[i].id = -1;

    declare_parameter("value_max", 10000);
    declare_parameter("value_inc", 40);
    declare_parameter("value_dec", 10);
    declare_parameter("value_fail_dec", 1);
    declare_parameter("value_match_limit", 100);
    declare_parameter("fail_dec_min", 100);
    declare_parameter("feedback_inc", 1000);
    declare_parameter("feedback_dec", 500);
    declare_parameter("pos_reinforce_inc", 10);
    declare_parameter("pos_reinforce_max", 100);
    declare_parameter("pub_timeout", 200);
    declare_parameter("vis_per_pub", 1);
    declare_parameter("pos_reinforce_timeout", 500);
    declare_parameter("uncertainty_limit", 40);
    declare_parameter("require_recent_visual", false);
    declare_parameter("visual_confirm_timeout_ms", 250);
    declare_parameter("min_visual_confirmations", 3);
    declare_parameter("assignment_hold_ms", 800);
    declare_parameter("assignment_follow_track_hold", true);
    declare_parameter("infantry_assignment_stickiness_bonus", 220);
    declare_parameter("infantry_dominance_lock_margin", 600);
    declare_parameter("slot_switch_guard_ms", 1200);
    declare_parameter("slot_switch_confirmations", 3);
    declare_parameter("slot_switch_score_margin", 500);
    declare_parameter("same_color_min_separation", 0.9);
    declare_parameter("async_identity_fusion", true);
    declare_parameter("identity_hold_ms", 2500);
    declare_parameter("identity_stale_value_dec", 1);
    auto default_team_color = declare_parameter("default_team_color", std::string("blue"));
    std::transform(default_team_color.begin(), default_team_color.end(), default_team_color.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    color = default_team_color == "red" ? radar_interface::team_color::C_RED : radar_interface::team_color::C_BLUE;
    std::vector<std::string> img_ns = declare_parameter("img_ns", std::vector<std::string> {"/radar"});

    target_sub = this->create_subscription<radar_interface::msg::TargetArray>(
        "pc_detector/targets", rclcpp::SystemDefaultsQoS(), std::bind(&MatcherNode::target_callback, this, std::placeholders::_1));
    // detected_target_sub = this->create_subscription<radar_interface::msg::DetectedTargetArray>(
    //     "img_recognizer/detected_targets", rclcpp::SystemDefaultsQoS(), std::bind(&MatcherNode::detected_target_callback, this, std::placeholders::_1));
    for (const auto& img : img_ns)
        detected_target_subs.push_back(this->create_subscription<radar_interface::msg::DetectedTargetArray>(
            img + "/img_recognizer/detected_targets", rclcpp::SystemDefaultsQoS(), std::bind(&MatcherNode::detected_target_callback, this, std::placeholders::_1)));
    feedback_sub = this->create_subscription<radar_interface::msg::FeedbackTargetArray>(
        "matcher/feedback", rclcpp::SystemDefaultsQoS(), std::bind(&MatcherNode::feedback_callback, this, std::placeholders::_1));
    team_color_sub = this->create_subscription<radar_interface::team_color::msg>(
        "judge/color", rclcpp::SystemDefaultsQoS(), std::bind(&MatcherNode::team_color_callback, this, std::placeholders::_1));
    match_result_pub = this->create_publisher<radar_interface::msg::MatchResult>("matcher/match_result", rclcpp::SystemDefaultsQoS());
    vis_pub = this->create_publisher<sensor_msgs::msg::Image>("matcher/visualization", rclcpp::SystemDefaultsQoS());
    vis_timer = this->create_wall_timer(std::chrono::milliseconds(get_parameter("pub_timeout").as_int()), std::bind(&MatcherNode::vis_timer_callback, this));
    // pos_reinforce_timer = this->create_wall_timer(std::chrono::milliseconds(get_parameter("pos_reinforce_timeout").as_int()), std::bind(&MatcherNode::pos_reinforce_timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "target_matcher node started.");
}

void MatcherNode::team_color_callback(const radar_interface::team_color::msg& msg)
{
    color = static_cast<radar_interface::team_color::ENUM>(msg.data);
}

void MatcherNode::target_callback(const radar_interface::msg::TargetArray::SharedPtr msg)
{
    last_targets = *msg;
    // 用于存储正在跟踪的目标
    std::unordered_set<unsigned> tracked_targets;
    for (const auto& target : msg->targets)
        tracked_targets.insert(target.id);
    // 将当前的 map 中没有被跟踪的目标删除
    for (auto it = targets_value_map.begin(); it != targets_value_map.end();) {
        // RCLCPP_INFO(this->get_logger(), "Checking result_visualizertarget %u.", it->first);
        if (tracked_targets.find(it->first) == tracked_targets.end()) {
            RCLCPP_INFO(this->get_logger(), "Target %ld removed.", it->first);
            target_last_visual_confirmed.erase(it->first);
            target_last_strict_idx.erase(it->first);
            target_visual_confirm_count.erase(it->first);
            target_last_published_idx.erase(it->first);
            for (auto& state : blue_slot_switch_states) {
                if (state.valid && state.candidate_id == it->first)
                    state.valid = false, state.candidate_id = -1, state.confirmations = 0;
            }
            for (auto& state : red_slot_switch_states) {
                if (state.valid && state.candidate_id == it->first)
                    state.valid = false, state.candidate_id = -1, state.confirmations = 0;
            }
            it = targets_value_map.erase(it);
        }
        else
            ++it;
    }
    // 将当前的目标添加到 map 中
    for (const auto& target : msg->targets)
    {
        if (targets_value_map.find(target.id) == targets_value_map.end()) {
            RCLCPP_INFO(this->get_logger(), "Target %lu added.", target.id);
            targets_value_map[target.id] = { 0 };
        }
    }
    match_and_pub(msg);
}

void MatcherNode::detected_target_callback(const radar_interface::msg::DetectedTargetArray::SharedPtr msg)
{
    auto fail_dec = [&](long target_id, ValueArray& value) { // 没有识别成功的
        if (should_preserve_identity_on_visual_miss(target_id))
            return;
        unsigned value_num = 0;
        for (auto v : value)
            if (v > 0)
                ++value_num;
        int min_value = value_num > 1 ? 0 : get_parameter("fail_dec_min").as_int();
        int dec = get_parameter("value_fail_dec").as_int();
        if (get_parameter("async_identity_fusion").as_bool())
            dec = std::min(dec, static_cast<int>(get_parameter("identity_stale_value_dec").as_int()));
        dec = std::max(dec, 0);
        for (auto& v : value)
            v = std::max(std::min(int(v), min_value), int(v - dec));
    };
    std::unordered_set<unsigned> checked_ids; // 存放已被检查过的 id
    // 遍历所有检测到的目标
    for (const auto& detected_target : msg->targets) {
        checked_ids.insert(detected_target.target.id);
        // 找到 targets_value_map 中对应的目标
        auto it = targets_value_map.find(detected_target.target.id);
        if (it != targets_value_map.end())
        {
            auto& value = it->second;
            int idx = to_idx(detected_target.color, detected_target.type);
            if (idx != -1) {
                target_last_visual_confirmed[detected_target.target.id] = this->now();
                auto idx_it = target_last_strict_idx.find(detected_target.target.id);
                if (idx_it != target_last_strict_idx.end() && idx_it->second == idx)
                    ++target_visual_confirm_count[detected_target.target.id];
                else {
                    target_last_strict_idx[detected_target.target.id] = idx;
                    target_visual_confirm_count[detected_target.target.id] = 1;
                }
            }
            if (idx == -1) {    // 检测到的目标颜色或类型未知
                if (should_preserve_identity_on_visual_miss(detected_target.target.id))
                    continue;
                // int reinforce_idx = pos_reinforce(detected_target.target.position[0], detected_target.target.position[1]);
                // if (reinforce_idx != -1 && value[reinforce_idx] > 0) {
                //     idx = reinforce_idx;
                // } else {
                    auto max_it = std::max_element(value.begin(), value.end());
                    long max_value = *max_it;
                    if (max_value > 0) {
                        int best_idx = std::distance(value.begin(), max_it);
                        int keep_dec = std::max<int>(1, static_cast<int>(get_parameter("value_fail_dec").as_int()));
                        int other_dec = std::max<int>(1, static_cast<int>(get_parameter("value_dec").as_int() / 4));
                        for (int i = 0; i < 12; ++i) {
                            int dec = i == best_idx ? keep_dec : other_dec;
                            value[i] = std::max(0l, long(value[i] - dec));
                        }
                        continue;
                    } else {
                        fail_dec(detected_target.target.id, value);
                        continue;
                    }
                // }
            }
            // 增强被识别到的类型，削弱其他类型
            int value_inc = get_parameter("value_inc").as_int();
            int value_dec = get_parameter("value_dec").as_int();
            value[idx] = std::min(value[idx] + value_inc, get_parameter("value_max").as_int());
            for (int i = 0; i < 12; ++i) {
                if (i != idx)
                    value[i] = std::max(0, int(value[i] - value_dec));
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Detected target %lu not found in targets_value_map.", detected_target.target.id);
        }
    }
    for (auto& [id, value] : targets_value_map) {
        if (checked_ids.find(id) == checked_ids.end())
            fail_dec(id, value);
    }
}

void MatcherNode::feedback_callback(const radar_interface::msg::FeedbackTargetArray::SharedPtr msg)
{
    for (const auto& feedback_target : msg->targets) {
        auto it = targets_value_map.find(feedback_target.id);
        if (it != targets_value_map.end()) {
            auto& value = it->second;
            int idx = to_idx(feedback_target.color, feedback_target.type);
            if (idx == -1) {
                RCLCPP_WARN(this->get_logger(), "Feedback target %lu has unknown color or type.", feedback_target.id);
                continue;
            }
            if (feedback_target.is_right)
                value[idx] = std::min(value[idx] + get_parameter("feedback_inc").as_int(), get_parameter("value_max").as_int());
            else
                value[idx] = std::max(value[idx] - get_parameter("feedback_dec").as_int(), 0l);
        } else {
            RCLCPP_WARN(this->get_logger(), "Feedback target %lu not found in targets_value_map.", feedback_target.id);
        }
    }
}

bool MatcherNode::should_preserve_identity_on_visual_miss(long target_id) const
{
    if (!get_parameter("async_identity_fusion").as_bool())
        return false;

    auto it = target_last_visual_confirmed.find(target_id);
    if (it == target_last_visual_confirmed.end())
        return false;

    const int hold_ms = get_parameter("identity_hold_ms").as_int();
    if (hold_ms < 0)
        return true;

    return (this->now() - it->second).nanoseconds() <= static_cast<int64_t>(hold_ms) * 1000000ll;
}

void MatcherNode::match_and_pub(const radar_interface::msg::TargetArray::SharedPtr msg)
{
    // RCLCPP_INFO(this->get_logger(), "Matching %lu targets.", msg->targets.size());

    // auto print_mat = [](const dlib::matrix<long>& mat) {
    //     std::string print_str = "Matrix: (" + std::to_string(mat.nr()) + ", " + std::to_string(mat.nc()) + ")";
    //     for (unsigned i = 0; i < mat.nr(); ++i) {
    //         print_str += "\n";
    //         for (unsigned j = 0; j < mat.nc(); ++j)
    //             print_str += std::to_string(mat(i, j)) + ", ";
    //     }
    //     RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "%s", print_str.c_str());
    // };

    // 构造矩阵 (目标在 msg 中的编号, 12 个兵result种)
    // 如果目标个数大于 12, 扩展矩阵
    unsigned size = std::max(msg->targets.size(), 12ul);
    dlib::matrix<long> cost_matrix(size, size);
    // 不是方阵似乎有 bug
    int value_match_limit = get_parameter("value_match_limit").as_int();
    int infantry_assignment_stickiness_bonus = get_parameter("infantry_assignment_stickiness_bonus").as_int();
    int infantry_dominance_lock_margin = get_parameter("infantry_dominance_lock_margin").as_int();
    for (unsigned t = 0; t < msg->targets.size(); ++t)
    {
        auto& target = msg->targets[t];
        auto it = targets_value_map.find(target.id);
        if (it != targets_value_map.end())
        {
            auto& value = it->second;
            for (unsigned i = 0; i < 12; ++i)
                cost_matrix(t, i) = value[i] >= value_match_limit ? value[i] : 0;
            auto last_pub_it = target_last_published_idx.find(target.id);
            if (last_pub_it != target_last_published_idx.end() && is_infantry_idx(last_pub_it->second)) {
                int last_idx = last_pub_it->second;
                if (last_idx >= 0 && last_idx < 12 && cost_matrix(t, last_idx) > 0)
                    cost_matrix(t, last_idx) += infantry_assignment_stickiness_bonus;
            }
            if (infantry_dominance_lock_margin > 0) {
                for (const auto& range : { std::pair<int, int> { 3, 5 }, std::pair<int, int> { 9, 11 } }) {
                    auto [best_idx, second_idx] = dominant_infantry_idx(value, range.first, range.second);
                    if (cost_matrix(t, best_idx) <= 0)
                        continue;
                    if (cost_matrix(t, best_idx) - cost_matrix(t, second_idx) < infantry_dominance_lock_margin)
                        continue;
                    for (int idx = range.first; idx <= range.second; ++idx) {
                        if (idx != best_idx)
                            cost_matrix(t, idx) = 0;
                    }
                }
            }
            for (unsigned i = 12; i < size; ++i)
                cost_matrix(t, i) = 0;
        }
        else
        {
            for (unsigned i = 0; i < 12; ++i)
                cost_matrix(t, i) = 0;
        }
    }
    for (unsigned t = msg->targets.size(); t < size; ++t)
    {
        for (unsigned i = 0; i < size; ++i)
            cost_matrix(t, i) = 0;
    }
    // print_mat(cost_matrix);

    // 结果 index 是目标编号, value 是兵种编号
    auto assignment = dlib::max_cost_assignment(cost_matrix);
    // std::string print_str = "Assignment: ";
    // for (auto a : assignment)
    //     print_str += std::to_string(a) + ", ";
    // print_str += "Cost: " + std::to_string(dlib::assignment_cost(cost_matrix, assignment));
    // RCLCPP_INFO(this->get_logger(), "%s", print_str.c_str());

    for (unsigned i = 0; i < assignment.size(); ++i) {
        if (assignment[i] >= 12)
            assignment[i] = -1;
        else if (i >= msg->targets.size())
            assignment[i] = -2;
        else if (targets_value_map.find(msg->targets[i].id) == targets_value_map.end())
            assignment[i] = -3;
        else if (targets_value_map[msg->targets[i].id][assignment[i]] == 0)
            assignment[i] = -4;
    }
    // print_str = "Assignment Edited: ";
    // for (auto a : assignment)
    //     print_str += std::to_string(a) + ", ";
    // RCLCPP_INFO(this->get_logger(), "%s", print_str.c_str());

    radar_interface::msg::MatchResult next_result;
    for (unsigned i = 0; i < 6; ++i)
        next_result.red[i].id = -1, next_result.blue[i].id = -1;

    const auto now = this->now();
    const auto hold_ns = static_cast<int64_t>(get_parameter("assignment_hold_ms").as_int()) * 1000000ll;
    const bool follow_track_hold = get_parameter("assignment_follow_track_hold").as_bool();
    const auto switch_guard_ns = static_cast<int64_t>(get_parameter("slot_switch_guard_ms").as_int()) * 1000000ll;
    const int switch_confirmations_required = get_parameter("slot_switch_confirmations").as_int();
    const int switch_score_margin = get_parameter("slot_switch_score_margin").as_int();
    const double same_color_min_sep = get_parameter("same_color_min_separation").as_double();
    const double same_color_min_sep_sq = same_color_min_sep * same_color_min_sep;
    const int uncertainty_limit = get_parameter("uncertainty_limit").as_int();

    std::vector<unsigned> process_order;
    process_order.reserve(assignment.size());
    for (unsigned i = 0; i < assignment.size(); ++i) {
        if (assignment[i] >= 0)
            process_order.push_back(i);
    }
    std::sort(process_order.begin(), process_order.end(), [this, msg, &assignment](unsigned a, unsigned b) {
        auto ita = targets_value_map.find(msg->targets[a].id);
        auto itb = targets_value_map.find(msg->targets[b].id);
        long sa = (ita == targets_value_map.end()) ? 0 : ita->second[assignment[a]];
        long sb = (itb == targets_value_map.end()) ? 0 : itb->second[assignment[b]];
        return sa > sb;
    });

    struct PublishedTarget {
        bool is_blue;
        std::array<double, 2> position;
    };
    std::vector<PublishedTarget> published_targets;
    published_targets.reserve(12);

    for (unsigned i : process_order) {
        if (assignment[i] >= 0) {
            auto& target = msg->targets[i];
            if (target.uncertainty >= get_parameter("uncertainty_limit").as_int()) {
                RCLCPP_DEBUG(this->get_logger(), "Target %ld filtered out: uncertainty %u >= limit %ld", target.id, target.uncertainty, get_parameter("uncertainty_limit").as_int());
                continue;
            }
            if (get_parameter("require_recent_visual").as_bool()
                && !get_parameter("async_identity_fusion").as_bool()) {
                auto it_visual = target_last_visual_confirmed.find(target.id);
                if (it_visual == target_last_visual_confirmed.end())
                    continue;
                const auto timeout_ms = get_parameter("visual_confirm_timeout_ms").as_int();
                if ((this->now() - it_visual->second).nanoseconds() > static_cast<int64_t>(timeout_ms) * 1000000ll)
                    continue;
                auto it_idx = target_last_strict_idx.find(target.id);
                auto it_count = target_visual_confirm_count.find(target.id);
                if (it_idx == target_last_strict_idx.end() || it_count == target_visual_confirm_count.end())
                    continue;
                if (it_idx->second != assignment[i])
                    continue;
                if (it_count->second < get_parameter("min_visual_confirmations").as_int())
                    continue;
            }
            auto value_it = targets_value_map.find(target.id);
            if (value_it == targets_value_map.end())
                continue;
            auto& value_array = value_it->second;
            if (infantry_dominance_lock_margin > 0 && is_infantry_idx(assignment[i])) {
                auto [begin, end] = infantry_idx_range(assignment[i]);
                auto [best_idx, second_idx] = dominant_infantry_idx(value_array, begin, end);
                if (best_idx != assignment[i]
                    && value_array[best_idx] - value_array[assignment[i]] >= infantry_dominance_lock_margin)
                    continue;
            }
            const bool is_blue = assignment[i] < 6;
            if (same_color_min_sep_sq > 0.0) {
                bool near_duplicate = false;
                for (const auto& published : published_targets) {
                    if (published.is_blue != is_blue)
                        continue;
                    const double dx = static_cast<double>(published.position[0] - target.position[0]);
                    const double dy = static_cast<double>(published.position[1] - target.position[1]);
                    if (dx * dx + dy * dy < same_color_min_sep_sq) {
                        near_duplicate = true;
                        break;
                    }
                }
                if (near_duplicate)
                    continue;
            }
            const int slot_idx = is_blue ? assignment[i] : assignment[i] - 6;
            const int absolute_idx = assignment[i];
            auto& held_targets = is_blue ? held_blue_targets : held_red_targets;
            auto& switch_states = is_blue ? blue_slot_switch_states : red_slot_switch_states;
            auto& held_target = held_targets[slot_idx];
            auto& switch_state = switch_states[slot_idx];
            if (held_target.valid && held_target.target.id != -1
                && (now - held_target.stamp).nanoseconds() <= hold_ns
                && held_target.target.id != target.id) {
                long incumbent_score = 0;
                auto incumbent_it = targets_value_map.find(held_target.target.id);
                if (incumbent_it != targets_value_map.end())
                    incumbent_score = incumbent_it->second[absolute_idx];
                const long challenger_score = value_array[absolute_idx];
                const bool challenger_stronger = challenger_score >= incumbent_score + switch_score_margin;

                if (!switch_state.valid || switch_state.candidate_id != target.id) {
                    switch_state.valid = true;
                    switch_state.candidate_id = target.id;
                    switch_state.first_seen = now;
                    switch_state.confirmations = 1;
                } else {
                    switch_state.confirmations++;
                }

                const bool guard_elapsed = (now - switch_state.first_seen).nanoseconds() >= switch_guard_ns;
                const bool enough_confirmations = switch_state.confirmations >= switch_confirmations_required;
                if (!challenger_stronger || !guard_elapsed || !enough_confirmations) {
                    RCLCPP_DEBUG(this->get_logger(),
                        "Protect slot %d from switching %ld -> %ld (incumbent=%ld challenger=%ld conf=%d/%d elapsed_ms=%.1f)",
                        absolute_idx,
                        held_target.target.id,
                        target.id,
                        incumbent_score,
                        challenger_score,
                        switch_state.confirmations,
                        switch_confirmations_required,
                        (now - switch_state.first_seen).nanoseconds() / 1e6);
                    continue;
                }
            } else {
                switch_state.valid = false;
                switch_state.candidate_id = -1;
                switch_state.confirmations = 0;
            }
            if (assignment[i] < 6) {
                next_result.blue[assignment[i]].id = target.id;
                next_result.blue[assignment[i]].position = target.position;
                target_last_published_idx[target.id] = assignment[i];
                held_blue_targets[assignment[i]].target = next_result.blue[assignment[i]];
                held_blue_targets[assignment[i]].stamp = now;
                held_blue_targets[assignment[i]].valid = true;
                blue_slot_switch_states[assignment[i]].valid = false;
                blue_slot_switch_states[assignment[i]].candidate_id = -1;
                blue_slot_switch_states[assignment[i]].confirmations = 0;
                published_targets.push_back(PublishedTarget { true, target.position });
                RCLCPP_INFO(this->get_logger(), "Assignment: target_id=%ld -> BLUE[type=%d], blue_scores=[%ld,%ld,%ld,%ld,%ld,%ld], red_scores=[%ld,%ld,%ld,%ld,%ld,%ld], pos=(%.2f,%.2f)", target.id, (int)assignment[i], value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], value_array[6], value_array[7], value_array[8], value_array[9], value_array[10], value_array[11], target.position[0], target.position[1]);
            } else {
                next_result.red[assignment[i] - 6].id = target.id;
                next_result.red[assignment[i] - 6].position = target.position;
                target_last_published_idx[target.id] = assignment[i];
                held_red_targets[assignment[i] - 6].target = next_result.red[assignment[i] - 6];
                held_red_targets[assignment[i] - 6].stamp = now;
                held_red_targets[assignment[i] - 6].valid = true;
                red_slot_switch_states[assignment[i] - 6].valid = false;
                red_slot_switch_states[assignment[i] - 6].candidate_id = -1;
                red_slot_switch_states[assignment[i] - 6].confirmations = 0;
                published_targets.push_back(PublishedTarget { false, target.position });
                RCLCPP_INFO(this->get_logger(), "Assignment: target_id=%ld -> RED[type=%d], blue_scores=[%ld,%ld,%ld,%ld,%ld,%ld], red_scores=[%ld,%ld,%ld,%ld,%ld,%ld], pos=(%.2f,%.2f)", target.id, (int)(assignment[i] - 6), value_array[0], value_array[1], value_array[2], value_array[3], value_array[4], value_array[5], value_array[6], value_array[7], value_array[8], value_array[9], value_array[10], value_array[11], target.position[0], target.position[1]);
            }
        }
    }

    for (unsigned i = 0; i < 6; ++i) {
        auto refresh_held_from_track = [&](auto& held_target) {
            if (!follow_track_hold || !held_target.valid || held_target.target.id == -1)
                return;
            for (const auto& track : msg->targets) {
                if (static_cast<long>(track.id) != held_target.target.id)
                    continue;
                if (static_cast<int>(track.uncertainty) >= uncertainty_limit)
                    return;
                held_target.target.position = track.position;
                return;
            }
        };
        if (next_result.blue[i].id == -1 && held_blue_targets[i].valid) {
            refresh_held_from_track(held_blue_targets[i]);
            if ((now - held_blue_targets[i].stamp).nanoseconds() <= hold_ns)
                next_result.blue[i] = held_blue_targets[i].target;
        }
        if (next_result.red[i].id == -1 && held_red_targets[i].valid) {
            refresh_held_from_track(held_red_targets[i]);
            if ((now - held_red_targets[i].stamp).nanoseconds() <= hold_ns)
                next_result.red[i] = held_red_targets[i].target;
        }
    }

    result = next_result;
}

void MatcherNode::vis_timer_callback()
{
    static int count = 0;
    if (count >= get_parameter("vis_per_pub").as_int()) {
        sensor_msgs::msg::Image::SharedPtr img;
        Visualization visualization;
        cv::Mat canvas = visualization.draw(targets_value_map, result, get_parameter("value_max").as_int());
        img = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", canvas).toImageMsg();
        vis_pub->publish(*img);
        count = 0;
    }
    match_result_pub->publish(result);
    ++count;
}
