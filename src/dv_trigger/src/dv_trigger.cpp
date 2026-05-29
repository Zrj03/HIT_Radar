#include <memory>
#include <dv_trigger/dv_trigger.hpp>



DvTriggerNode::DvTriggerNode()
    : rclcpp::Node("dv_trigger")
{
    dv_context.used_chances = 0;
    dv_context.now_chances = 0;
    dv_context.is_dv_trigered = false;
    dv_context.waiting_for_check = false;

    declare_parameter("dv_check_time", 5);
    declare_parameter("dv_max", 2);

    pub_custom_info = create_publisher<std_msgs::msg::String>("judge/custom_info", rclcpp::SystemDefaultsQoS());
    pub_radar_cmd = create_publisher<std_msgs::msg::UInt8>("judge/radar_cmd", rclcpp::SystemDefaultsQoS());
    
    sub_radar_info = create_subscription<radar_interface::msg::RadarInfo>("judge/radar_info", rclcpp::SystemDefaultsQoS(), std::bind(&DvTriggerNode::radar_info_callback, this, std::placeholders::_1));
    sub_team_color = create_subscription<radar_interface::team_color::msg>("judge/color", rclcpp::SystemDefaultsQoS(), std::bind(&DvTriggerNode::color_callback, this, std::placeholders::_1));
    sub_time = create_subscription<std_msgs::msg::UInt16>("judge/remain_time", rclcpp::SystemDefaultsQoS(), std::bind(&DvTriggerNode::time_callback, this, std::placeholders::_1));
}

bool DvTriggerNode::dv_available()
{
    return dv_context.waiting_for_check == 0 && !dv_context.is_dv_trigered &&
           dv_context.now_chances > dv_context.used_chances &&
           dv_context.used_chances < get_parameter("dv_max").as_int();
}

bool DvTriggerNode::trigger_dv(const std::string_view& reason)
{
    if (!dv_available()) {
        RCLCPP_WARN(
            get_logger(),
            "Trigger Failed. reason=%s, chances=%u, used=%u, triggered=%s, waiting=%u",
            reason.data(),
            dv_context.now_chances,
            dv_context.used_chances,
            dv_context.is_dv_trigered ? "true" : "false",
            dv_context.waiting_for_check);
        return false;
    }
    auto radar_cmd = std_msgs::msg::UInt8();
    radar_cmd.data = std::min(long(dv_context.used_chances + 1), get_parameter("dv_max").as_int());
    dv_context.waiting_for_check = get_parameter("dv_check_time").as_int();
    pub_radar_cmd->publish(radar_cmd);

    auto message = std_msgs::msg::String();
    message.data = "【触发】双倍易伤：";
    message.data += reason;
    pub_custom_info->publish(message);
    RCLCPP_INFO(get_logger(), "Trigger double vulnerability: %d / %d, reason: %s", dv_context.used_chances, dv_context.now_chances, reason.data());
    return true;
}

void DvTriggerNode::radar_info_callback(const radar_interface::msg::RadarInfo& info){
    dv_context.now_chances = info.dv_chances;
    dv_context.is_dv_trigered = info.dv_triggered;

    if (dv_context.waiting_for_check > 0) {
        if (dv_context.is_dv_trigered) {
            dv_context.waiting_for_check = 0;
            ++dv_context.used_chances;
            RCLCPP_INFO(get_logger(), "Double vulnerability successfully triggered");
        }
        else
            --dv_context.waiting_for_check;
    }
    if (dv_context.used_chances > info.dv_chances) {
        dv_context.used_chances = info.dv_chances;
        RCLCPP_WARN(get_logger(), "Double vulnerability has wrong response");
    }
    static bool already_pub = false;
    if (!dv_context.is_dv_trigered && dv_context.now_chances > dv_context.used_chances) {
        if (!already_pub) {
            already_pub = true;
            auto message = std_msgs::msg::String();
            message.data = "【可用】双倍易伤";
            pub_custom_info->publish(message);
        }
        trigger_dv("裁判系统判定已获得双倍易伤机会");
    } else
        already_pub = false;
}

void DvTriggerNode::color_callback(const radar_interface::team_color::msg& color_)
{
    radar_interface::team_color::ENUM last_color = color;
    color = static_cast<radar_interface::team_color::ENUM>(color_.data);
    if (last_color != color)
        switch (color) {
        case radar_interface::team_color::C_RED:
            RCLCPP_INFO(get_logger(), "WE ARE <<<RED>>>");
            break;
        case radar_interface::team_color::C_BLUE:
            RCLCPP_INFO(get_logger(), "WE ARE <<<BLUE>>>");
            break;
        default:
            RCLCPP_WARN(get_logger(), "Unknow the radar id");
            break;
        }
}

void DvTriggerNode::time_callback(const std_msgs::msg::UInt16& time)
{
    static uint16_t last_time = 0;
    if (time.data > last_time || time.data > 400)
        dv_context.used_chances = 0;    // 重置 trick
    if (time.data > 0 && time.data <= 120)
        trigger_dv("比赛剩余2分钟仍有双倍易伤机会");
    last_time = time.data;
}
