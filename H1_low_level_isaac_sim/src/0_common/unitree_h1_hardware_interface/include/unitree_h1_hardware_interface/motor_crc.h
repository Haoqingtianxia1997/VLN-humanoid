#ifndef _MOTOR_CRC_H_
#define _MOTOR_CRC_H_

#include <stdint.h>
#include <array>
#include "rclcpp/rclcpp.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/motor_cmd.hpp"

constexpr int HIGHLEVEL = 0xee;
constexpr int LOWLEVEL = 0xff;
constexpr int TRIGERLEVEL = 0xf0;
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// joint index
enum H1_Joint_Index
{
    LEFT_HIP_YAW,
    LEFT_HIP_ROLL,
    LEFT_HIP_PITCH,
    LEFT_KNEE,
    LEFT_ANKLE,

    RIGHT_HIP_YAW,
    RIGHT_HIP_ROLL,
    RIGHT_HIP_PITCH,
    RIGHT_KNEE,
    RIGHT_ANKLE,
    
    TORSO,

    LEFT_SHOULDER_PITCH,
    LEFT_SHOULDER_ROLL,
    LEFT_SHOULDER_YAW,
    LEFT_ELBOW,
    
    RIGHT_SHOULDER_PITCH,
    RIGHT_SHOULDER_ROLL,
    RIGHT_SHOULDER_YAW,
    RIGHT_ELBOW
};
        

uint32_t crc32_core(uint32_t* ptr, uint32_t len);
void get_crc(unitree_hg::msg::LowCmd& msg);

#endif