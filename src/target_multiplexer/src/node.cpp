#include "target_multiplexer/target_multiplexer.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <toml.hpp>
#include <algorithm>
#include <cctype>

// 由于裁判系统反馈是 1Hz ，如果要猜测一个目标，应该要在 1s 内发送 2 次以上的坐标，所以应该要记住每次发送的

using namespace target_multiplexer;

MultiplexerNode::MultiplexerNode()
    : rclcpp::Node("multiplexer")
{
    for (int i = 0; i < 6; ++i) {
        last_match_result.red[i].id = -1;
        last_match_result.blue[i].id = -1;
        // Without judge feedback, default to NONE to avoid skipping all slots.
        full_high_light[i] = FULL_HIGHLIGHT_STATUS::NONE;
    }
    last_mark.mark_progress = 0;
    last_pub_id.fill(-1);
    // load_blind_guess();

    declare_parameter("double_send_thres", 100);
    declare_parameter("detected_hold_sec", 1.5);
    declare_parameter("hold_guard_dist", 1.2);
    auto default_team_color = declare_parameter("default_team_color", std::string("red"));
    std::transform(default_team_color.begin(), default_team_color.end(), default_team_color.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (default_team_color == "blue")
        color = team_color::C_BLUE;
    else
        color = team_color::C_RED;
    RCLCPP_INFO(get_logger(), "Default team color: %s", color == team_color::C_BLUE ? "BLUE" : "RED");
    robot_num_per_team = 6;
    robot_num = robot_num_per_team * 2;

    match_result_sub = create_subscription<radar_interface::msg::MatchResult>("rm_radar_pipeline/match_result",
        rclcpp::SystemDefaultsQoS(), [this](const radar_interface::msg::MatchResult& msg) {last_match_result = msg;});
    detected_sub = create_subscription<radar_interface::msg::TargetArray>("rm_radar_pipeline/targets",
        rclcpp::SystemDefaultsQoS(), [this](const radar_interface::msg::TargetArray& msg) {
            if (!msg.targets.empty()) {
                last_detected = msg;

                return;
            }

                last_detected = msg;
        });
    radar_mark_sub = create_subscription<radar_interface::msg::RadarMarkData>("judge/radar_mark_data",
        rclcpp::SystemDefaultsQoS(), std::bind(&MultiplexerNode::radar_mark_callback, this, std::placeholders::_1));
    team_color_sub = create_subscription<radar_interface::team_color::msg>("judge/color",
        rclcpp::SystemDefaultsQoS(), std::bind(&MultiplexerNode::team_color_callback, this, std::placeholders::_1));

    feedback_pub = create_publisher<radar_interface::msg::FeedbackTargetArray>("rm_radar_pipeline/feedback", rclcpp::SystemDefaultsQoS());

    multiplexer_timer = create_wall_timer(std::chrono::milliseconds(declare_parameter("match_send_interval", 80)),
        std::bind(&MultiplexerNode::multiplexer, this));
}

void MultiplexerNode::team_color_callback(const radar_interface::team_color::msg& msg)
{
    team_color last_color = color;
    color = static_cast<team_color>(msg.data);
    if (last_color != color)
        switch (color) {
        case team_color::C_RED:
            RCLCPP_INFO(get_logger(), "WE ARE <<<RED>>>");
            break;
        case team_color::C_BLUE:
            RCLCPP_INFO(get_logger(), "WE ARE <<<BLUE>>>");
            break;
        default:
            RCLCPP_WARN(get_logger(), "Unknow the radar id");
            break;
        }
}

// void MultiplexerNode::load_blind_guess()
// {
//     auto data = toml::parse(
//         std::filesystem::path(ament_index_cpp::get_package_share_directory("target_multiplexer")) / "config" / "blind_guess.toml");

//     auto& robots = data["robot"];
//     for (unsigned i = 0; i < 6; ++i) {
//         if (!robots.contains(std::to_string(i)))
//             continue;
//         auto& robot = robots[std::to_string(i)];
//         for (auto& pos : toml::get<std::vector<std::vector<float>>>(robot["positions"]))
//             blind_guess[i].emplace_back(pos[0], pos[1]);
//     }
// }
