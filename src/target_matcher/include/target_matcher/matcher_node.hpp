#pragma once

#include <map>
#include <array>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <radar_interface/msg/target_array.hpp>
#include <radar_interface/msg/detected_target_array.hpp>
#include <radar_interface/msg/match_result.hpp>
#include <radar_interface/msg/feedback_target_array.hpp>
#include <radar_interface/team_color.hpp>

class MatcherNode : public rclcpp::Node {
public:
    using ValueArray = std::array<long, 12>;
    using TargetValueMap = std::map<long, ValueArray>;

private:
    struct HeldMatchedTarget {
        radar_interface::msg::MatchedTarget target;
        rclcpp::Time stamp;
        bool valid = false;
    };

    struct SlotSwitchState {
        long candidate_id = -1;
        rclcpp::Time first_seen;
        int confirmations = 0;
        bool valid = false;
    };

    rclcpp::Subscription<radar_interface::msg::TargetArray>::SharedPtr target_sub;
    std::vector<rclcpp::Subscription<radar_interface::msg::DetectedTargetArray>::SharedPtr> detected_target_subs;
    rclcpp::Subscription<radar_interface::msg::FeedbackTargetArray>::SharedPtr feedback_sub;
    rclcpp::Subscription<radar_interface::team_color::msg>::SharedPtr team_color_sub;

    rclcpp::Publisher<radar_interface::msg::MatchResult>::SharedPtr match_result_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub;
    rclcpp::TimerBase::SharedPtr vis_timer;
    rclcpp::TimerBase::SharedPtr pos_reinforce_timer;

    TargetValueMap targets_value_map;
    std::map<long, rclcpp::Time> target_last_visual_confirmed;
    std::map<long, int> target_last_strict_idx;
    std::map<long, int> target_visual_confirm_count;
    std::map<long, int> target_last_published_idx;
    radar_interface::msg::MatchResult result;
    std::array<HeldMatchedTarget, 6> held_blue_targets;
    std::array<HeldMatchedTarget, 6> held_red_targets;
    std::array<SlotSwitchState, 6> blue_slot_switch_states;
    std::array<SlotSwitchState, 6> red_slot_switch_states;
    radar_interface::msg::TargetArray last_targets;
    radar_interface::team_color::ENUM color = radar_interface::team_color::C_BLUE;

    void target_callback(const radar_interface::msg::TargetArray::SharedPtr msg);
    void detected_target_callback(const radar_interface::msg::DetectedTargetArray::SharedPtr msg);
    void feedback_callback(const radar_interface::msg::FeedbackTargetArray::SharedPtr msg);
    void team_color_callback(const radar_interface::team_color::msg& msg);
    void match_and_pub(const radar_interface::msg::TargetArray::SharedPtr msg);
    void vis_timer_callback();
    void pos_reinforce_timer_callback();
    bool should_preserve_identity_on_visual_miss(long target_id) const;
    bool is_enemy_slot(bool slot_is_blue) const;
    bool is_in_enemy_lost_mark_rect(const radar_interface::msg::MatchedTarget& target) const;

    int pos_reinforce(float x, float y);
    int inner_pos_reinforce(float x, float y);

public:
    MatcherNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
};
