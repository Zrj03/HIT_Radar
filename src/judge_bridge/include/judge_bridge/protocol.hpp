#pragma once

#include <atomic>
#include <cstdint>
#include <boost/endian.hpp>

#pragma pack(1)
struct frame_header_t {
    uint8_t sof;
    uint16_t data_length;
    uint8_t package_sequence;
    uint8_t CRC8;
};

// 定义于裁判系统手册
struct radar_mark_data_t {
    uint16_t mark_progress;
};

struct radar_info_t {
    uint8_t radar_info;
};

struct radar_cmd_t {
    uint8_t radar_cmd;
    uint8_t password_cmd;
    uint8_t password_1;
    uint8_t password_2;
    uint8_t password_3;
    uint8_t password_4;
    uint8_t password_5;
    uint8_t password_6;
};

struct map_robot_data_t {
    uint16_t opponent_hero_position_x;
    uint16_t opponent_hero_position_y;
    uint16_t opponent_engineer_position_x;
    uint16_t opponent_engineer_position_y;
    uint16_t opponent_infantry_3_position_x;
    uint16_t opponent_infantry_3_position_y;
    uint16_t opponent_infantry_4_position_x;
    uint16_t opponent_infantry_4_position_y;
    uint16_t opponent_aerial_position_x;
    uint16_t opponent_aerial_position_y;
    uint16_t opponent_sentry_position_x;
    uint16_t opponent_sentry_position_y;
    uint16_t ally_hero_position_x;
    uint16_t ally_hero_position_y;
    uint16_t ally_engineer_position_x;
    uint16_t ally_engineer_position_y;
    uint16_t ally_infantry_3_position_x;
    uint16_t ally_infantry_3_position_y;
    uint16_t ally_infantry_4_position_x;
    uint16_t ally_infantry_4_position_y;
    uint16_t ally_aerial_position_x;
    uint16_t ally_aerial_position_y;
    uint16_t ally_sentry_position_x;
    uint16_t ally_sentry_position_y;
};

struct radar_link_position_t {
    uint16_t opponent_hero_position_x;
    uint16_t opponent_hero_position_y;
    uint16_t opponent_engineer_position_x;
    uint16_t opponent_engineer_position_y;
    uint16_t opponent_infantry_3_position_x;
    uint16_t opponent_infantry_3_position_y;
    uint16_t opponent_infantry_4_position_x;
    uint16_t opponent_infantry_4_position_y;
    uint16_t opponent_aerial_position_x;
    uint16_t opponent_aerial_position_y;
    uint16_t opponent_sentry_position_x;
    uint16_t opponent_sentry_position_y;
};

struct radar_link_hp_t {
    uint16_t opponent_hero_hp;
    uint16_t opponent_engineer_hp;
    uint16_t opponent_infantry_3_hp;
    uint16_t opponent_infantry_4_hp;
    uint16_t reserved;
    uint16_t opponent_sentry_hp;
};

struct radar_link_bullet_t {
    uint16_t opponent_hero_bullet;
    uint16_t opponent_infantry_3_bullet;
    uint16_t opponent_infantry_4_bullet;
    uint16_t opponent_aerial_bullet;
    uint16_t opponent_sentry_bullet;
};

struct radar_link_coin_and_occupy_t {
    uint16_t opponent_remaining_coin;
    uint16_t opponent_total_coin;
    uint32_t occupy_status;
};

struct radar_link_buff_t {
    uint8_t opponent_hero_hp_recover_buff;
    uint16_t opponent_hero_shooter_cooling_buff;
    uint8_t opponent_hero_defense_buff;
    uint8_t opponent_hero_negative_defense_buff;
    uint16_t opponent_hero_attack_buff;
    uint8_t opponent_engineer_hp_recover_buff;
    uint16_t opponent_engineer_shooter_cooling_buff;
    uint8_t opponent_engineer_defense_buff;
    uint8_t opponent_engineer_negative_defense_buff;
    uint16_t opponent_engineer_attack_buff;
    uint8_t opponent_infantry_3_hp_recover_buff;
    uint16_t opponent_infantry_3_shooter_cooling_buff;
    uint8_t opponent_infantry_3_defense_buff;
    uint8_t opponent_infantry_3_negative_defense_buff;
    uint16_t opponent_infantry_3_attack_buff;
    uint8_t opponent_infantry_4_hp_recover_buff;
    uint16_t opponent_infantry_4_shooter_cooling_buff;
    uint8_t opponent_infantry_4_defense_buff;
    uint8_t opponent_infantry_4_negative_defense_buff;
    uint16_t opponent_infantry_4_attack_buff;
    uint8_t opponent_sentry_hp_recover_buff;
    uint16_t opponent_sentry_shooter_cooling_buff;
    uint8_t opponent_sentry_defense_buff;
    uint8_t opponent_sentry_negative_defense_buff;
    uint16_t opponent_sentry_attack_buff;
    uint8_t opponent_sentry_current_pose;
};

struct radar_link_password_t {
    uint8_t password[6];
};

// 2023
// struct map_command_t
// {
//     float target_position_x;
//     float target_position_y;
//     float target_position_z;
//     uint8_t cmd_keyboard;
//     uint16_t target_robot_id;
// };
// 2024
struct map_command_t {
    float target_position_x;
    float target_position_y;
    uint8_t cmd_keyboard;
    uint8_t target_robot_id;
    uint8_t cmd_source;
};

struct custom_info_t
{ 
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[30];
};

struct robot_status_t
{
    uint8_t robot_id;
    uint8_t robot_level;
    uint16_t current_HP; 
    uint16_t maximum_HP;
    uint16_t shooter_barrel_cooling_value;
    uint16_t shooter_barrel_heat_limit;
    uint16_t chassis_power_limit; 

    uint8_t power_management_gimbal_output : 1;
    uint8_t power_management_chassis_output : 1; 
    uint8_t power_management_shooter_output : 1;
};

enum INTERACTION_CMD {
    RADAR_CMD = 0x0121,
    SENTRY_DATA = 0x0201,
    MAP_KEYBOARD = 0x0202,
    UWB_DATA = 0x0203,
    SENTRY_TARGETS = 0x0205,
};

struct robot_interaction_header_t {
    uint16_t data_cmd_id;
    uint16_t sender_id;
    uint16_t receiver_id;
};

struct robot_interaction_dv_data_t {
    // uint16_t data_cmd_id = RADAR_CMD;
    // uint16_t sender_id;
    // uint16_t receiver_id = 0x8080;
    robot_interaction_header_t header;
    radar_cmd_t cmd;
};

struct robot_interaction_map_data_t {
    // 0x0202
    robot_interaction_header_t header;
    map_command_t map_cmd;
};

struct robot_interaction_map_robot_data_t {
    robot_interaction_header_t header;
    map_robot_data_t map_robot_data;
};

struct robot_interaction_sentry_map_robot_data_t {
    robot_interaction_header_t header;
    struct robot_map_observation {
        int16_t robot_id;           // recognized armor type/id, -1: unknown
        int16_t hp;                 // -1: unknown
        int16_t pos_x;              // cm, -1: unknown
        int16_t pos_y;              // cm, -1: unknown
    } robots[10];                   // opponent/ally: hero, engineer, infantry3, infantry4, sentry
    bool enemy_outpost_alive;        // true: alive, false: destroyed
};

struct robot_interaction_uwb_t {
    float hero_x;
    float hero_y;
    float engineer_x;
    float engineer_y;
    float standard_3_x;
    float standard_3_y;
    float standard_4_x;
    float standard_4_y;
    float standard_5_x;
    float standard_5_y;
};

struct robot_interaction_sentry_targets_t {
    robot_interaction_header_t header;
};

struct robot_interaction_sentry_target_t {
    float x;
    float y;
};

static_assert(sizeof(radar_mark_data_t) == 2);
static_assert(sizeof(radar_cmd_t) == 8);
static_assert(sizeof(map_robot_data_t) == 48);
static_assert(sizeof(radar_link_position_t) == 24);
static_assert(sizeof(radar_link_hp_t) == 12);
static_assert(sizeof(radar_link_bullet_t) == 10);
static_assert(sizeof(radar_link_coin_and_occupy_t) == 8);
static_assert(sizeof(radar_link_buff_t) == 36);
static_assert(sizeof(radar_link_password_t) == 6);
static_assert(sizeof(robot_interaction_map_robot_data_t) == 54);
static_assert(sizeof(robot_interaction_sentry_map_robot_data_t) == 87);
static_assert(sizeof(robot_interaction_sentry_targets_t) == 6);
static_assert(sizeof(robot_interaction_sentry_target_t) == 8);

struct game_status_t
{ 
    uint8_t game_type_and_progress; 
    uint16_t stage_remain_time; 
    uint64_t SyncTimeStamp; 
}; 

struct game_robot_HP_t
{ 
    uint16_t ally_1_robot_HP;
    uint16_t ally_2_robot_HP;
    uint16_t ally_3_robot_HP;
    uint16_t ally_4_robot_HP;
    uint16_t reserved;
    uint16_t ally_7_robot_HP;
    uint16_t ally_outpost_HP;
    uint16_t ally_base_HP;
}; 

static_assert(sizeof(game_robot_HP_t) == 16);

#pragma pack()

// 具体十六进制码参见裁判系统手册
enum CMD_ID {
    GAME_STATUS = 0x0001,
    GAME_ROBOT_HP = 0x0003,
    ROBOT_STATUS = 0x0201,
    DETECT_PROCESS = 0x020C,
    RADAR_INFO = 0x020E,
    ROBOT_MAP = 0x0305,
    MAP_COMMAND = 0x0303,
    INTERACTION_DATA = 0x0301,
    SEND_CUSTOM_INFO = 0x0308,
    RADAR_LINK_POSITION = 0x0A01,
    RADAR_LINK_HP = 0x0A02,
    RADAR_LINK_BULLET = 0x0A03,
    RADAR_LINK_COIN_AND_OCCUPY = 0x0A04,
    RADAR_LINK_BUFF = 0x0A05,
    RADAR_LINK_PASSWORD = 0x0A06,
};

enum RADAR_ID {
    R_RED = 9,
    R_BLUE = 109,
};

// BLUE, RED
constexpr uint8_t SENTRY_ID[] = {107, 7};
constexpr uint8_t HERO_ID[] = {101, 1};
constexpr uint8_t ENGINEER_ID[] = {102, 2};
constexpr uint8_t STANDARD_1_ID[] = {103, 3};
constexpr uint8_t STANDARD_2_ID[] = {104, 4};
constexpr uint8_t AERIAL_ID[] = {106, 6};

constexpr uint8_t RED_ROBOT[] = {7, 1, 2, 3, 4, 6};
constexpr uint8_t BLUE_ROBOT[] = {107, 101, 102, 103, 104, 106};


// blue, red
constexpr uint16_t AERIAL_CLIENT[2] = { 0x016A, 0x0106, };
constexpr uint16_t RADAR_ID[2] = { 109, 9, };
