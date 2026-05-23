#pragma once

#include "judge_bridge/protocol.hpp"

#include <cstddef>

#include <radar_interface/msg/detail/map_command__struct.hpp>
#include <radar_interface/msg/detail/radar_info__struct.hpp>
#include <radar_interface/msg/detail/radar_mark_data__struct.hpp>
#include <radar_interface/msg/radar_info.hpp>
#include <radar_interface/msg/radar_mark_data.hpp>
#include <radar_interface/msg/radar_link_position.hpp>
#include <radar_interface/msg/radar_link_hp.hpp>
#include <radar_interface/msg/radar_link_bullet.hpp>
#include <radar_interface/msg/radar_link_coin_and_occupy.hpp>
#include <radar_interface/msg/radar_link_buff.hpp>
#include <radar_interface/msg/radar_link_password.hpp>
#include <radar_interface/msg/map_command.hpp>


inline radar_interface::msg::RadarInfo decode_radar_info(const radar_info_t& ori) {
    radar_interface::msg::RadarInfo rtn;
    rtn.dv_chances = ori.radar_info & 0x3;
    rtn.dv_triggered = (ori.radar_info >> 2) & 1;
    return rtn;
}

inline radar_interface::msg::RadarMarkData decode_radar_mark_data(const radar_mark_data_t& ori){
    radar_interface::msg::RadarMarkData rtn;
    rtn.mark_progress = ori.mark_progress;
    return rtn;
}

inline radar_interface::msg::RadarLinkPosition decode_radar_link_position(const radar_link_position_t& ori) {
    radar_interface::msg::RadarLinkPosition rtn;
    rtn.opponent_hero_position_x = ori.opponent_hero_position_x;
    rtn.opponent_hero_position_y = ori.opponent_hero_position_y;
    rtn.opponent_engineer_position_x = ori.opponent_engineer_position_x;
    rtn.opponent_engineer_position_y = ori.opponent_engineer_position_y;
    rtn.opponent_infantry_3_position_x = ori.opponent_infantry_3_position_x;
    rtn.opponent_infantry_3_position_y = ori.opponent_infantry_3_position_y;
    rtn.opponent_infantry_4_position_x = ori.opponent_infantry_4_position_x;
    rtn.opponent_infantry_4_position_y = ori.opponent_infantry_4_position_y;
    rtn.opponent_aerial_position_x = ori.opponent_aerial_position_x;
    rtn.opponent_aerial_position_y = ori.opponent_aerial_position_y;
    rtn.opponent_sentry_position_x = ori.opponent_sentry_position_x;
    rtn.opponent_sentry_position_y = ori.opponent_sentry_position_y;
    return rtn;
}

inline radar_interface::msg::RadarLinkHp decode_radar_link_hp(const radar_link_hp_t& ori) {
    radar_interface::msg::RadarLinkHp rtn;
    rtn.opponent_hero_hp = ori.opponent_hero_hp;
    rtn.opponent_engineer_hp = ori.opponent_engineer_hp;
    rtn.opponent_infantry_3_hp = ori.opponent_infantry_3_hp;
    rtn.opponent_infantry_4_hp = ori.opponent_infantry_4_hp;
    rtn.reserved = ori.reserved;
    rtn.opponent_sentry_hp = ori.opponent_sentry_hp;
    return rtn;
}

inline radar_interface::msg::RadarLinkBullet decode_radar_link_bullet(const radar_link_bullet_t& ori) {
    radar_interface::msg::RadarLinkBullet rtn;
    rtn.opponent_hero_bullet = ori.opponent_hero_bullet;
    rtn.opponent_infantry_3_bullet = ori.opponent_infantry_3_bullet;
    rtn.opponent_infantry_4_bullet = ori.opponent_infantry_4_bullet;
    rtn.opponent_aerial_bullet = ori.opponent_aerial_bullet;
    rtn.opponent_sentry_bullet = ori.opponent_sentry_bullet;
    return rtn;
}

inline radar_interface::msg::RadarLinkCoinAndOccupy decode_radar_link_coin_and_occupy(const radar_link_coin_and_occupy_t& ori) {
    radar_interface::msg::RadarLinkCoinAndOccupy rtn;
    rtn.opponent_remaining_coin = ori.opponent_remaining_coin;
    rtn.opponent_total_coin = ori.opponent_total_coin;
    rtn.occupy_status = ori.occupy_status;
    return rtn;
}

inline radar_interface::msg::RadarLinkBuff decode_radar_link_buff(const radar_link_buff_t& ori) {
    radar_interface::msg::RadarLinkBuff rtn;
    rtn.opponent_hero_hp_recover_buff = ori.opponent_hero_hp_recover_buff;
    rtn.opponent_hero_shooter_cooling_buff = ori.opponent_hero_shooter_cooling_buff;
    rtn.opponent_hero_defense_buff = ori.opponent_hero_defense_buff;
    rtn.opponent_hero_negative_defense_buff = ori.opponent_hero_negative_defense_buff;
    rtn.opponent_hero_attack_buff = ori.opponent_hero_attack_buff;
    rtn.opponent_engineer_hp_recover_buff = ori.opponent_engineer_hp_recover_buff;
    rtn.opponent_engineer_shooter_cooling_buff = ori.opponent_engineer_shooter_cooling_buff;
    rtn.opponent_engineer_defense_buff = ori.opponent_engineer_defense_buff;
    rtn.opponent_engineer_negative_defense_buff = ori.opponent_engineer_negative_defense_buff;
    rtn.opponent_engineer_attack_buff = ori.opponent_engineer_attack_buff;
    rtn.opponent_infantry_3_hp_recover_buff = ori.opponent_infantry_3_hp_recover_buff;
    rtn.opponent_infantry_3_shooter_cooling_buff = ori.opponent_infantry_3_shooter_cooling_buff;
    rtn.opponent_infantry_3_defense_buff = ori.opponent_infantry_3_defense_buff;
    rtn.opponent_infantry_3_negative_defense_buff = ori.opponent_infantry_3_negative_defense_buff;
    rtn.opponent_infantry_3_attack_buff = ori.opponent_infantry_3_attack_buff;
    rtn.opponent_infantry_4_hp_recover_buff = ori.opponent_infantry_4_hp_recover_buff;
    rtn.opponent_infantry_4_shooter_cooling_buff = ori.opponent_infantry_4_shooter_cooling_buff;
    rtn.opponent_infantry_4_defense_buff = ori.opponent_infantry_4_defense_buff;
    rtn.opponent_infantry_4_negative_defense_buff = ori.opponent_infantry_4_negative_defense_buff;
    rtn.opponent_infantry_4_attack_buff = ori.opponent_infantry_4_attack_buff;
    rtn.opponent_sentry_hp_recover_buff = ori.opponent_sentry_hp_recover_buff;
    rtn.opponent_sentry_shooter_cooling_buff = ori.opponent_sentry_shooter_cooling_buff;
    rtn.opponent_sentry_defense_buff = ori.opponent_sentry_defense_buff;
    rtn.opponent_sentry_negative_defense_buff = ori.opponent_sentry_negative_defense_buff;
    rtn.opponent_sentry_attack_buff = ori.opponent_sentry_attack_buff;
    rtn.opponent_sentry_current_pose = ori.opponent_sentry_current_pose;
    return rtn;
}

inline radar_interface::msg::RadarLinkPassword decode_radar_link_password(const radar_link_password_t& ori) {
    radar_interface::msg::RadarLinkPassword rtn;
    for (size_t i = 0; i < rtn.password.size(); ++i)
        rtn.password[i] = ori.password[i];
    return rtn;
}
