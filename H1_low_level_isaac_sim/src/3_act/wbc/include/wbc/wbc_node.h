#ifndef WBC_NODE
#define WBC_NODE

#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "wbc_interface/msg/wbc_reference.hpp"
#include "whole_body_controller.h"


class WBCNode : public rclcpp::Node {
private:
    // Members
    std::unique_ptr<WholeBodyController> wbc_;
    WBCConfig config_;

    // ROS Communications
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<wbc_interface::msg::WBCReference>::SharedPtr mpc_ref_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // Robot State and References
    std::mutex state_mutex_;
    Eigen::VectorXd q_current_;
    Eigen::VectorXd v_current_;

    std::mutex ref_mutex_;
    WBCReferenceData mpc_refs_;

    bool has_state_ = false;
    bool has_refs_ = false;

    // Node settings
    double update_rate_hz_;

public:
    WBCNode();
    ~WBCNode();
    
private:
    // ROS 2 parameter handling
    void declare_parameters();
    void load_parameters();
    void log_parameters();
    rcl_interfaces::msg::SetParametersResult on_parameter_update(
        const std::vector<rclcpp::Parameter>& params);

    // Callbacks
    void control_loop_callback();
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void mpc_ref_callback(const wbc_interface::msg::WBCReference::SharedPtr msg);
};

#endif //WBC_NODE