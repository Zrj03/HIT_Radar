#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/subscription_base.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/u_int16.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <stdexcept>

#include "judge_bridge/protocol.hpp"
#include "judge_bridge/serial.hpp"
#include "judge_bridge/crc.hpp"

#include <radar_interface/msg/radar_info.hpp>
#include <radar_interface/msg/radar_cmd.hpp>
#include <radar_interface/msg/radar_mark_data.hpp>
#include <radar_interface/msg/radar_link_position.hpp>
#include <radar_interface/msg/radar_link_hp.hpp>
#include <radar_interface/msg/radar_link_bullet.hpp>
#include <radar_interface/msg/radar_link_coin_and_occupy.hpp>
#include <radar_interface/msg/radar_link_buff.hpp>
#include <radar_interface/msg/radar_link_password.hpp>
#include <radar_interface/msg/map_command.hpp>
#include <radar_interface/msg/armor.hpp>
#include <radar_interface/msg/detected_target_array.hpp>
#include <radar_interface/msg/match_result.hpp>
#include <radar_interface/msg/matched_target.hpp>
#include <radar_interface/msg/target_array.hpp>
#include <radar_interface/msg/game_robot_hp.hpp>
#include <radar_interface/msg/uwb_data.hpp>
#include <radar_interface/team_color.hpp>


using namespace JudgeBridge;
using namespace radar_interface;

class JudgeBridgeNode : public rclcpp::Node
{
private:
    std::unique_ptr<JudgeSerial> judge_serial;
    rclcpp::Publisher<radar_interface::msg::RadarMarkData>::SharedPtr pub_radar_mark_data;
    rclcpp::Publisher<radar_interface::msg::RadarInfo>::SharedPtr pub_radar_info;
    rclcpp::Publisher<radar_interface::msg::RadarLinkPosition>::SharedPtr pub_radar_link_position;
    rclcpp::Publisher<radar_interface::msg::RadarLinkHp>::SharedPtr pub_radar_link_hp;
    rclcpp::Publisher<radar_interface::msg::RadarLinkBullet>::SharedPtr pub_radar_link_bullet;
    rclcpp::Publisher<radar_interface::msg::RadarLinkCoinAndOccupy>::SharedPtr pub_radar_link_coin_and_occupy;
    rclcpp::Publisher<radar_interface::msg::RadarLinkBuff>::SharedPtr pub_radar_link_buff;
    rclcpp::Publisher<radar_interface::msg::RadarLinkPassword>::SharedPtr pub_radar_link_password;
    rclcpp::Publisher<radar_interface::team_color::msg>::SharedPtr pub_color;
    rclcpp::Publisher<std_msgs::msg::UInt16>::SharedPtr pub_remain_time;
    rclcpp::Publisher<radar_interface::msg::GameRobotHP>::SharedPtr pub_game_robot_hp;
    rclcpp::Publisher<radar_interface::msg::MapCommand>::SharedPtr pub_map_keyboard;
    rclcpp::Publisher<radar_interface::msg::UwbData>::SharedPtr pub_uwb_data;
    rclcpp::Publisher<radar_interface::msg::TargetArray>::SharedPtr pub_sentry_targets;

    std::thread read_thread;

    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr sub_radar_cmd;
    rclcpp::Subscription<radar_interface::msg::MatchResult>::SharedPtr sub_match_result;
    rclcpp::Subscription<radar_interface::msg::MatchResult>::SharedPtr sub_match_result_for_map;
    rclcpp::Subscription<radar_interface::msg::DetectedTargetArray>::SharedPtr sub_detected_targets;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_enemy_outpost_alive;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr sub_uav_target;
    std::atomic<team_color::ENUM> color { team_color::UNKNOWN };
    std::atomic_bool has_enemy_outpost_visual_state { false };
    std::atomic_bool enemy_outpost_alive { false };
    bool has_last_enemy_outpost_custom_info { false };
    bool last_enemy_outpost_custom_info_alive { false };
    rclcpp::Time last_enemy_outpost_custom_info_time {};

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_custom_info;
    static constexpr int16_t UNKNOWN_OBS_VALUE = -1;
    static constexpr int16_t POSITION_SCALE = 100;

    bool has_game_status { false };
    bool has_game_robot_hp { false };
    int16_t remaining_steps { UNKNOWN_OBS_VALUE };
    std::array<int16_t, 6> red_robot_hp {
        UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE,
        UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE };
    std::array<int16_t, 6> blue_robot_hp {
        UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE,
        UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE, UNKNOWN_OBS_VALUE };
    int16_t red_outpost_hp { UNKNOWN_OBS_VALUE };
    int16_t red_base_hp { UNKNOWN_OBS_VALUE };
    int16_t blue_outpost_hp { UNKNOWN_OBS_VALUE };
    int16_t blue_base_hp { UNKNOWN_OBS_VALUE };
    struct SentryObservationSlot {
        bool recognized { false };
        int16_t robot_id { UNKNOWN_OBS_VALUE };
        double pos_x { 0.0 };
        double pos_y { 0.0 };
    };
    static constexpr size_t SENTRY_OBSERVATION_SLOT_COUNT = 12;
    static constexpr size_t SENTRY_OBSERVATION_ALLY_OFFSET = 6;
    static constexpr size_t SENTRY_PACKET_ROBOT_COUNT = 10;
    static constexpr size_t SENTRY_PACKET_ALLY_OFFSET = 5;
    std::array<SentryObservationSlot, SENTRY_OBSERVATION_SLOT_COUNT> sentry_observation_slots {};
    bool has_sentry_targets_packet { false };
    uint16_t latest_sentry_target_count { 0 };
    mutable std::mutex sentry_observation_mutex;
    mutable std::mutex uav_target_mutex;
    bool has_uav_target { false };
    geometry_msgs::msg::PointStamped latest_uav_target;
    bool has_last_sentry_send_time { false };
    rclcpp::Time last_sentry_send_time {};

    void init_serial();
    void filter_handler(JudgeSerial::JudgePair message);
    map_robot_data_t build_map_robot_data(const radar_interface::msg::MatchResult& topic_message) const;
    void send_map_robot_data(const radar_interface::msg::MatchResult& topic_message);
    robot_interaction_map_robot_data_t build_sentry_map_robot_data(const map_robot_data_t& map_robot_data) const;
    void write_sentry_map_robot_data(const map_robot_data_t& map_robot_data);
    robot_interaction_sentry_map_robot_data_t build_sentry_map_observation_data(
        INTERACTION_CMD data_cmd_id = INTERACTION_CMD::SENTRY_DATA) const;
    void write_sentry_map_observation_data();

    void send_custom_info(const std::string& str);
    void map_command_callback(const map_command_t& cmd);
    void robot_status_callback(const robot_status_t& status);
    void game_status_callback(const game_status_t& status);
    void game_robot_hp_callback(const game_robot_HP_t& hp);
    void interaction_data_callback(const std::vector<uint8_t>& data);
    void sentry_targets_callback(
        const robot_interaction_sentry_targets_t& header,
        const std::vector<robot_interaction_sentry_target_t>& targets);
    void detected_targets_callback(const radar_interface::msg::DetectedTargetArray& detected_targets);
    void enemy_outpost_alive_callback(const std_msgs::msg::Bool& msg);
    void uav_target_callback(const geometry_msgs::msg::PointStamped& msg);

#if 0
    void check_enemy_invasion(const radar_interface::msg::MatchResult& msg);
    void send_invasion_alert(const radar_interface::msg::MatchResult& msg);
    void check_enemy_watch_zone(const radar_interface::msg::MatchResult& msg);
    void send_enemy_watch_zone_alert(const radar_interface::msg::MatchResult& msg);
#endif

    void send_sentry_data(const radar_interface::msg::MatchResult& topic_message);
    void send_radar_cmd(const std_msgs::msg::UInt8 &radar_cmd);

public:
    JudgeBridgeNode();
};
