#ifndef ACADOS_MPC_NODE_H
#define ACADOS_MPC_NODE_H

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <rclcpp/rclcpp.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <mutex>

// Include Messages
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "mpc_interface/msg/y_reference.hpp"
#include "wbc_interface/msg/wbc_reference.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

// Include acados headers
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados/ocp_nlp/ocp_nlp_sqp_rti.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_srbm_robot.h"
#include "blasfeo_d_aux_ext_dep.h"

class AcadosMPCNode : public rclcpp::Node {
private:
    // --- Configuration Struct ---
    // Holds all tunable parameters for the SRBM MPC
    struct MPCConfig {
        double mass;
        double gravity;

        std::vector<double> w_q; // State weights (Diagonal of Q matrix)
        std::vector<double> w_r; // Control weights (Diagonal of R matrix)
        double force_max; // Constraint limits
    };

    // --- ROS 2 Communications ---
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr state_sub_;
    rclcpp::Subscription<mpc_interface::msg::YReference>::SharedPtr reference_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr foot_pose_sub_;
    rclcpp::Publisher<wbc_interface::msg::WBCReference>::SharedPtr wbc_ref_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    rclcpp::TimerBase::SharedPtr timer_;

    // --- Acados Solver Members ---
    srbm_robot_solver_capsule* ocp_capsule_;
    ocp_nlp_config* ocp_nlp_config_;
    ocp_nlp_dims* ocp_nlp_dims_;
    ocp_nlp_in* ocp_nlp_in_;
    ocp_nlp_out* ocp_nlp_out_;
    void* ocp_nlp_opts_;
    
    // --- Member Variables ---
    MPCConfig config_;
    double current_state_[SRBM_ROBOT_NBX0];
    double parameter_[SRBM_ROBOT_NP];
    double yref_[SRBM_ROBOT_N][SRBM_ROBOT_NY];
    double yref_e_[SRBM_ROBOT_NY];
    std::mutex data_mutex_;

    bool state_received_;
    bool yref_received_;
    bool parameter_received_;

public:
    AcadosMPCNode();
    ~AcadosMPCNode();

private:
    // --- Core Methods ---
    void state_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void reference_callback(const mpc_interface::msg::YReference::SharedPtr msg);
    void parameter_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
    void control_loop_callback();
    void publish_solution();

    void initialize_ocp_solver();
    void update_ocp_parameters(double param[SRBM_ROBOT_NP]);
    void update_x0(double x0[SRBM_ROBOT_NBX0]);
    void update_yref(double yref[SRBM_ROBOT_N][SRBM_ROBOT_NY]);
    void update_yref_e(double yref_e[SRBM_ROBOT_NYN]);
    void update_solver_weights();
    void update_solver_constraints();

    // --- Parameter Handling Methods ---
    void declare_parameters();
    void load_parameters();
    void log_parameters();
    rcl_interfaces::msg::SetParametersResult on_parameter_update(const std::vector<rclcpp::Parameter>& params);

    // --- Helper & Utility Methods ---
    void get_input_at_stage(
        double u[SRBM_ROBOT_NU], 
        int stage
    ) ;
    void get_state_at_stage(
        double x[SRBM_ROBOT_NX], 
        int stage
    );
    void msg2state(
        double x0[SRBM_ROBOT_NX], 
        const nav_msgs::msg::Odometry::SharedPtr msg
    );
    void msg2param(
        double param[SRBM_ROBOT_NP], 
        const geometry_msgs::msg::PoseArray::SharedPtr msg
    );
    void msg2yref(
        double yref[SRBM_ROBOT_N][SRBM_ROBOT_NY], 
        double yref_e[SRBM_ROBOT_NYN], 
        const mpc_interface::msg::YReference::SharedPtr msg);
};

#endif  // ACADOS_MPC_NODE_H