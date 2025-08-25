#ifndef UNITREE_H1_HARDWARE_INTERFACE_HPP_
#define UNITREE_H1_HARDWARE_INTERFACE_HPP_

#include <vector>
#include <string>
#include <mutex>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "unitree_h1_hardware_interface/motor_crc_hg.h"

namespace unitree_h1_hardware
{
    
class UnitreeH1HardwareInterface : public hardware_interface::SystemInterface
{
private:
    // ROS 2 Kommunikation
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr cmd_pub_;
    rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr state_sub_;

    std::shared_ptr<unitree_hg::msg::LowState> last_state_msg_;
    std::mutex state_mutex_;

    rclcpp::Executor::SharedPtr executor_;
    std::thread executor_thread_;
    
    // Speicher für die Zustände und Befehle der Gelenke
    std::vector<double> hw_commands_;
    std::vector<double> hw_states_eff_;
    std::vector<double> hw_states_pos_;
    std::vector<double> hw_states_vel_;

    std::array<double, 4> hw_imu_orientation_;     // x, y, z, w
    std::array<double, 3> hw_imu_angular_velocity_; // x, y, z
    std::array<double, 3> hw_imu_linear_acceleration_; // x, y, z

    std::vector<int> urdf_to_actuator_map_;
    std::vector<int> urdf_to_sensor_joint_map_;
    mjModel* mj_model_ = nullptr;
    mjData* mj_data_ = nullptr;
public:
    RCLCPP_SHARED_PTR_DEFINITIONS(UnitreeH1HardwareInterface)
    ~UnitreeH1HardwareInterface() override;

    // Standard-Methoden von ros2_control
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

    // Die Kernmethoden: Lesen von der Simulation und Schreiben zur Simulation
    hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    
private:
    void create_urdf_xml_maps();

};

}  // namespace unitree_h1_hardware

#endif  // UNITREE_H1_HARDWARE_INTERFACE_HPP_