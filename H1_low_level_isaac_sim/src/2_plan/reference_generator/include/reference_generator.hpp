#ifndef REFERENCE_GENERATOR_H
#define REFERENCE_GENERATOR_H

#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#include "geometry_msgs/msg/pose.hpp"
#include "mpc/generated_acados/acados_solver_srbm_robot.h"

const double MPC_DT = 0.02;
const double ROBOT_MASS = 34.0;
const double GRAVITY = 9.81;


inline std::tuple<std::vector<std::array<double, SRBM_ROBOT_NY>>, std::array<double, SRBM_ROBOT_NYN>> slow_walk(
        const std::array<double, SRBM_ROBOT_NX>& initial_ref_state,
        double target_linear_velocity_x,
        double target_linear_velocity_y,
        double target_linear_velocity_z,
        double target_angular_velocity_z,
        double ground_time_s,
        double swing_time_s
) {
    // --- Stage Reference ---
    std::vector<std::array<double, SRBM_ROBOT_NY>> yref_trajectory;
    yref_trajectory.reserve(SRBM_ROBOT_N);

    std::array<double, SRBM_ROBOT_NY> current_yref_stage;
    current_yref_stage.fill(0.0); 

    for (int j = 0; j < SRBM_ROBOT_NX; ++j) {
        current_yref_stage[j] = initial_ref_state[j];
    }

    // Set initial target velocities within the yref vector
    current_yref_stage[7] = target_linear_velocity_x;
    current_yref_stage[8] = target_linear_velocity_y;
    current_yref_stage[9] = target_linear_velocity_z;
    current_yref_stage[12] = target_angular_velocity_z;


    // Gait timing parameters
    double half_step_cycle_s = ground_time_s + swing_time_s;
    double full_mass_force = ROBOT_MASS * GRAVITY;

    for (int i = 0; i < SRBM_ROBOT_N; ++i) {
        double current_time_s = i * MPC_DT;
        // This calculates the position within a half-step cycle
        double time_in_half_step_cycle = fmod(current_time_s, half_step_cycle_s);

        if (time_in_half_step_cycle < ground_time_s) {
            // Foot 1 is in stance, Foot 2 is in swing
            current_yref_stage[15] = full_mass_force; // f_c1.z (Foot 1) stance
            current_yref_stage[18] = 0.0;             // f_c2.z (Foot 2) swing
        } else {
            // Foot 1 is in swing, Foot 2 is in stance
            current_yref_stage[15] = 0.0;             // f_c1.z (Foot 1) swing
            current_yref_stage[18] = full_mass_force; // f_c2.z (Foot 2) stance
        }

        current_yref_stage[0] += target_linear_velocity_x * MPC_DT;
        current_yref_stage[1] += target_linear_velocity_y * MPC_DT;
        current_yref_stage[2] += target_linear_velocity_z * MPC_DT;

        yref_trajectory.push_back(current_yref_stage);
    }

    // --- Terminal Reference ---
    std::array<double, SRBM_ROBOT_NYN> yref_e;
    yref_e.fill(0.0);

    for (int j = 0; j < SRBM_ROBOT_NX; ++j) {
        yref_e[j] = current_yref_stage[j];
    }
    yref_e[0] += target_linear_velocity_x * MPC_DT;
    yref_e[1] += target_linear_velocity_y * MPC_DT;
    yref_e[2] += target_linear_velocity_z * MPC_DT;

    return {yref_trajectory, yref_e};
}

#endif // REFERENCE_GENERATOR_H