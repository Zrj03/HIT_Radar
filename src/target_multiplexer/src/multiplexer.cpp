#include "target_multiplexer/target_multiplexer.hpp"

using namespace target_multiplexer;

bool MultiplexerNode::has_nearby_detection(const HeldMapTarget& held_target, double dist_sqr_threshold) const
{
    if (last_detected.targets.empty()) {
        // If detector output is temporarily empty, avoid clearing all held targets at once.
        return true;
    }

    for (const auto& target : last_detected.targets) {
        const double dx = target.position[0] - held_target.position_x;
        const double dy = target.position[1] - held_target.position_y;
        if (dx * dx + dy * dy <= dist_sqr_threshold)
            return true;
    }
    return false;
}

uint16_t MultiplexerNode::mark_mask_for_type(unsigned type)
{
    constexpr uint16_t masks[6] = {
        radar_interface::msg::RadarMarkData::SENTRY_MASK,
        radar_interface::msg::RadarMarkData::HERO_MASK,
        radar_interface::msg::RadarMarkData::ENGINEER_MASK,
        radar_interface::msg::RadarMarkData::INFANTRY_3_MASK,
        radar_interface::msg::RadarMarkData::INFANTRY_4_MASK,
        radar_interface::msg::RadarMarkData::AERIAL_MASK,
    };
    return type < 6 ? masks[type] : 0;
}

bool MultiplexerNode::is_enemy_slot(int slot_idx) const
{
    const bool slot_is_blue = slot_idx >= robot_num_per_team;
    return color == team_color::C_RED ? slot_is_blue : !slot_is_blue;
}

bool MultiplexerNode::mark_set(const radar_interface::msg::RadarMarkData& mark, unsigned type)
{
    return (mark.mark_progress & mark_mask_for_type(type)) != 0;
}

radar_interface::msg::MatchedTarget MultiplexerNode::get_match_for_slot(int slot_idx) const
{
    const int team_idx = slot_idx % robot_num_per_team;
    const bool slot_is_blue = slot_idx >= robot_num_per_team;
    if (slot_is_blue)
        return last_match_result.blue[team_idx];
    return last_match_result.red[team_idx];
}

void MultiplexerNode::multiplexer()
{
#define NEXT                                         \
    redo_idx = redo_idx == -1 ? send_idx : redo_idx; \
    next_iter();                                     \
    goto redo;

    static bool double_send_sign = false;
    static int send_idx = 0;
    bool stop_iter = false;

    auto next_iter = [&]() {
        /// 轮换
        if (!stop_iter) {
            ++send_idx;
            if (send_idx >= robot_num)
                send_idx = 0;
        }
    };

    int redo_idx = -1;
redo:
    const int team_idx = send_idx % robot_num_per_team;
    const bool enemy_slot = is_enemy_slot(send_idx);

    if (redo_idx == send_idx) {
        // 轮了一遍没的发就退出
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Cannot multiplex output");
        if (enemy_slot && full_high_light[team_idx] == FULL_HIGHLIGHT_STATUS::HIGHLIGHT)  // 避免因为满高亮而不发送数据
            full_high_light[team_idx] = FULL_HIGHLIGHT_STATUS::SKIPPED;
        else
            return;
    }

    const auto now = this->now();
    const auto detected_hold_ns = static_cast<int64_t>(get_parameter("detected_hold_sec").as_double() * 1e9);
    const double hold_guard_dist = get_parameter("hold_guard_dist").as_double();
    const double hold_guard_dist_sqr = hold_guard_dist * hold_guard_dist;

    auto last_match = get_match_for_slot(send_idx);

    if (last_match.id != -1) {
        // 如果存在已匹配目标
        last_pub_id[send_idx] = last_match.id;
        held_map_targets[send_idx].position_x = last_match.position[0];
        held_map_targets[send_idx].position_y = last_match.position[1];
        held_map_targets[send_idx].stamp = now;
        held_map_targets[send_idx].valid = true;

        if (enemy_slot && full_high_light[team_idx] == FULL_HIGHLIGHT_STATUS::SKIPPED) {// 对于满高亮的不继续发送，留出带宽
            NEXT
        }

        if (double_send_sign)           // 已经两次发送，不再发送
            double_send_sign = false;
        // 判断如果进度很小就发两次
        else if (enemy_slot && !mark_set(last_mark, team_idx))
            double_send_sign = true, stop_iter = true;
    } else if (held_map_targets[send_idx].valid
        && (now - held_map_targets[send_idx].stamp).nanoseconds() <= detected_hold_ns
        && has_nearby_detection(held_map_targets[send_idx], hold_guard_dist_sqr)) {
        stop_iter = true;
    } else {
        held_map_targets[send_idx].valid = false;
        NEXT
    }

    next_iter();
}

void MultiplexerNode::radar_mark_callback(const radar_interface::msg::RadarMarkData& msg)
{
    radar_interface::msg::FeedbackTargetArray fb_array;

    team_color enemy_color;
    const int enemy_slot_offset = color == radar_interface::team_color::C_BLUE ? 0 : robot_num_per_team;
    switch (color) {
    case radar_interface::team_color::C_BLUE:
        enemy_color = team_color::C_RED;
        break;
    case radar_interface::team_color::C_RED:
        enemy_color = team_color::C_BLUE;
        break;
    default:
        RCLCPP_WARN(get_logger(), "Unknown color!");
        return;
    }

    for (int i = 0; i < robot_num_per_team; ++i) {
        const bool now_marked = mark_set(msg, i);
        const bool last_marked = mark_set(last_mark, i);
        const int enemy_slot = enemy_slot_offset + i;

        radar_interface::msg::FeedbackTarget fb;
        fb.id = last_pub_id[enemy_slot];
        fb.type = i;
        fb.color = enemy_color;

        if (now_marked && !last_marked) {
            fb.is_right = true;
            RCLCPP_INFO(get_logger(), "Right map: type: %d, id: %ld, mark_progress: %#x", i, last_pub_id[enemy_slot], msg.mark_progress);
            if (last_pub_id[enemy_slot] > -1)
                fb_array.targets.push_back(fb);
        }

        if (now_marked)
            full_high_light[i] = FULL_HIGHLIGHT_STATUS::HIGHLIGHT;
        else
            full_high_light[i] = FULL_HIGHLIGHT_STATUS::NONE;

        // if (last_pub_id[i] > -1)
        //     fb_array.targets.push_back(fb);
    }
    if (fb_array.targets.size() > 0)
        feedback_pub->publish(fb_array);
    last_mark = msg;
}
