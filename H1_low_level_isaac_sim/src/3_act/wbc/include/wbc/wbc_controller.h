#ifndef WBC_CONTROLLER_H
#define WBC_CONTROLLER_H

#include <vector>
#include <string>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "controller_interface/controller_interface.hpp"
#include "realtime_tools/realtime_tools/realtime_buffer.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

#include "wbc/whole_body_controller.h"
#include "wbc_interface/msg/wbc_reference.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include "wbc/wbc_parameters.hpp"

namespace wbc
{

class WbcController : public controller_interface::ControllerInterface
{
private:
    // Ihre Kern-Logik und Konfiguration
    std::unique_ptr<WholeBodyController> wbc_;

    // Vektoren für Zustände und Befehle
    Eigen::VectorXd q_current_;
    Eigen::VectorXd v_current_;

    // Namen der Gelenke aus der URDF/ros2_control Konfiguration
    std::vector<std::string> joint_names_;

    std::vector<size_t> command_interface_map_;
    std::vector<size_t> position_state_interface_map_;
    std::vector<size_t> velocity_state_interface_map_;

    // Subscriptions
    rclcpp::Subscription<wbc_interface::msg::WBCReference>::SharedPtr ref_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr foot_poses_pub_;

    // Echtzeit-sichere Buffer
    realtime_tools::RealtimeBuffer<std::shared_ptr<wbc_interface::msg::WBCReference>> rt_ref_buffer_;
    realtime_tools::RealtimeBuffer<std::shared_ptr<nav_msgs::msg::Odometry>> rt_odom_buffer_;

    // ROS-Parameter Handling
    std::shared_ptr<wbc::ParamListener> param_listener_;
    wbc::Params params_;

public:
    WbcController();

    // Die wichtigsten Lifecycle-Methoden von ros2_control
    controller_interface::CallbackReturn on_init() override;
    controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
    controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
    controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

    // Diese Methoden definieren, was Ihr Controller von der Hardware braucht und an sie sendet
    controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    controller_interface::InterfaceConfiguration state_interface_configuration() const override;

    // Dies ist Ihre neue "control_loop_callback"
    controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
    void update_parameters();
    bool extract_wbc_references(const wbc_interface::msg::WBCReference& msg, wbc::WBCReferenceData& refs_local);
    void read_state_from_hardware(const nav_msgs::msg::Odometry& odom_data);
    bool write_torques_to_hardware(const Eigen::VectorXd& torques);
    void publish_foot_poses();
    void create_hold_position_reference(wbc::WBCReferenceData& refs) const;
};

} // namespace wbc

#endif // WBC_CONTROLLER_H