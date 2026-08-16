#pragma once

#include <rclcpp/rclcpp.hpp>
#include <radar_interface/msg/match_result.hpp>
#include <radar_interface/msg/target_array.hpp>
#include <radar_interface/msg/radar_mark_data.hpp>
#include <radar_interface/msg/feedback_target_array.hpp>
#include <radar_interface/team_color.hpp>


namespace target_multiplexer {
using team_color = radar_interface::team_color::ENUM;

// 用于盲猜的编解码
constexpr size_t decode_idx(int64_t x) { return -x - 2; }
constexpr int64_t encode_idx(size_t x) { return -x - 2; }

enum class FULL_HIGHLIGHT_STATUS {
    NONE,
    HIGHLIGHT,
    SKIPPED
};

class MultiplexerNode : public rclcpp::Node {
private:
    struct HeldMapTarget {
        double position_x = 0.0;
        double position_y = 0.0;
        rclcpp::Time stamp;
        bool valid = false;
    };

    rclcpp::Subscription<radar_interface::msg::MatchResult>::SharedPtr match_result_sub;
    rclcpp::Subscription<radar_interface::msg::TargetArray>::SharedPtr detected_sub;
    rclcpp::Subscription<radar_interface::msg::RadarMarkData>::SharedPtr radar_mark_sub;
    rclcpp::Subscription<radar_interface::team_color::msg>::SharedPtr team_color_sub;

    rclcpp::Publisher<radar_interface::msg::FeedbackTargetArray>::SharedPtr feedback_pub;

    rclcpp::TimerBase::SharedPtr multiplexer_timer;

    radar_interface::msg::MatchResult last_match_result;
    radar_interface::msg::TargetArray last_detected;
    radar_interface::msg::RadarMarkData last_mark;

    std::array<int64_t, 12> last_pub_id;
    std::array<HeldMapTarget, 12> held_map_targets;
    std::array<FULL_HIGHLIGHT_STATUS, 6> full_high_light;

    // Default to C_RED to allow local testing without referee system connected
    team_color color = team_color::C_RED;
    int robot_num = 12;
    int robot_num_per_team = 6;

    void radar_mark_callback(const radar_interface::msg::RadarMarkData& msg);
    void team_color_callback(const radar_interface::team_color::msg& msg);
    bool has_nearby_detection(const HeldMapTarget& held_target, double dist_sqr_threshold) const;

    void multiplexer();

    static uint16_t mark_mask_for_type(unsigned type);
    bool is_enemy_slot(int slot_idx) const;
    static bool mark_set(const radar_interface::msg::RadarMarkData& mark, unsigned type);
    radar_interface::msg::MatchedTarget get_match_for_slot(int slot_idx) const;

public:
    MultiplexerNode();
};

}
