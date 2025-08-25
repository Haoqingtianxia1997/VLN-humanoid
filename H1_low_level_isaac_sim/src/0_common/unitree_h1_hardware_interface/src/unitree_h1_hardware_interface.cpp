#include "unitree_h1_hardware_interface/unitree_h1_hardware_interface.hpp"

#include "pluginlib/class_list_macros.hpp"

namespace unitree_h1_hardware
{

UnitreeH1HardwareInterface::~UnitreeH1HardwareInterface() { 
    RCLCPP_INFO(node_->get_logger(), "Destruction of UnitreeH1HardwareInterface...");
    if (mj_data_) {
        mj_deleteData(mj_data_);
        mj_data_ = nullptr;
    }
    if (mj_model_) {
        mj_deleteModel(mj_model_);
        mj_model_ = nullptr;
    }
}

hardware_interface::CallbackReturn UnitreeH1HardwareInterface::on_init(const hardware_interface::HardwareInfo & info) {
    if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS) {
        return hardware_interface::CallbackReturn::ERROR;
    }

    // --- Initialise the hardware interface ---
    info_ = info;
    node_ = rclcpp::Node::make_shared("unitree_h1_hardware_interface");
    
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);

    if (info_.joints.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "No joints defined in hardware info.");
        return hardware_interface::CallbackReturn::ERROR;
    }

    hw_commands_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_pos_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_vel_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
    hw_states_eff_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());

    // --- Load the MuJoCo model ---
    std::string model_path;
    try {
        model_path = info.hardware_parameters.at("mujoco_model_path");
    } catch (const std::out_of_range& ex) {
        RCLCPP_FATAL(node_->get_logger(), "Hardware parameter 'mujoco_model_path' not found in URDF!");
        return hardware_interface::CallbackReturn::ERROR;
    }

    char error[1000] = {0};
    mj_model_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
    if (!mj_model_) {
        RCLCPP_FATAL(node_->get_logger(), "Could not load MuJoCo model: %s", error);
        return hardware_interface::CallbackReturn::ERROR;
    }

    // --- Create MuJoCo data ---
    mj_data_ = mj_makeData(mj_model_);
    if (!mj_data_) {
        RCLCPP_FATAL(node_->get_logger(), "Could not create MuJoCo data structure.");
        mj_deleteModel(mj_model_); // Free the model
        mj_model_ = nullptr;
        return hardware_interface::CallbackReturn::ERROR;
    }

    create_urdf_xml_maps();

    RCLCPP_INFO(node_->get_logger(), "Hardware-Interface initialized for %ld joints.", info_.joints.size());
    return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> UnitreeH1HardwareInterface::export_state_interfaces() {
    std::vector<hardware_interface::StateInterface> state_interfaces;
    for (uint i = 0; i < info_.joints.size(); i++) {
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_pos_[i]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_states_vel_[i]));
        state_interfaces.emplace_back(hardware_interface::StateInterface(
            info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_states_eff_[i]));
    }

    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "orientation.x", &hw_imu_orientation_[0]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "orientation.y", &hw_imu_orientation_[1]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "orientation.z", &hw_imu_orientation_[2]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "orientation.w", &hw_imu_orientation_[3]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "angular_velocity.x", &hw_imu_angular_velocity_[0]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "angular_velocity.y", &hw_imu_angular_velocity_[1]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "angular_velocity.z", &hw_imu_angular_velocity_[2]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "linear_acceleration.x", &hw_imu_linear_acceleration_[0]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "linear_acceleration.y", &hw_imu_linear_acceleration_[1]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        "h1_imu", "linear_acceleration.z", &hw_imu_linear_acceleration_[2]));

    return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> UnitreeH1HardwareInterface::export_command_interfaces() {
    std::vector<hardware_interface::CommandInterface> command_interfaces;
    for (uint i = 0; i < info_.joints.size(); i++) {
        command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_commands_[i]));
    }
    return command_interfaces;
}

hardware_interface::CallbackReturn UnitreeH1HardwareInterface::on_activate(const rclcpp_lifecycle::State & /*previous_state*/) {
    RCLCPP_DEBUG(node_->get_logger(), "Activating Hardware Interface...");
    
    state_sub_ = node_->create_subscription<unitree_hg::msg::LowState>(
        "/lowstate", rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<unitree_hg::msg::LowState> msg) {
            RCLCPP_DEBUG_ONCE(node_->get_logger(), "Received first LowState message.");
            std::lock_guard<std::mutex> lock(state_mutex_);
            last_state_msg_ = msg;
        });

    cmd_pub_ = node_->create_publisher<unitree_hg::msg::LowCmd>(
        "/lowcmd", rclcpp::SystemDefaultsQoS());

    executor_thread_ = std::thread([this]() {
        executor_->spin();
    });

    RCLCPP_DEBUG(node_->get_logger(), "Hardware Interface successfully activated.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn UnitreeH1HardwareInterface::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/) {
    RCLCPP_DEBUG(node_->get_logger(), "Deactivating Hardware Interface.");
    executor_->cancel();
    if (executor_thread_.joinable()) {
        executor_thread_.join();
    }
    cmd_pub_.reset();
    state_sub_.reset();
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type UnitreeH1HardwareInterface::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
    // Kopiere die letzte Nachricht sicher
    std::shared_ptr<unitree_hg::msg::LowState> state_msg;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_msg = last_state_msg_;
    }

    if (!state_msg) {
        return hardware_interface::return_type::OK;
    }
    
    for (uint i = 0; i < info_.joints.size(); i++) {
        int motor_index = urdf_to_sensor_joint_map_[i];
        if (motor_index != -1) {
            const auto& motor = state_msg->motor_state[motor_index];
            hw_states_pos_[i] = motor.q;
            hw_states_vel_[i] = motor.dq;
            hw_states_eff_[i] = motor.tau_est;
        } else {
            RCLCPP_WARN(node_->get_logger(), "No actuator mapping found for joint '%s'.", info_.joints[i].name.c_str());
        }
    }

    hw_imu_orientation_[0] = state_msg->imu_state.quaternion[0];
    hw_imu_orientation_[1] = state_msg->imu_state.quaternion[1];
    hw_imu_orientation_[2] = state_msg->imu_state.quaternion[2];
    hw_imu_orientation_[3] = state_msg->imu_state.quaternion[3];
    hw_imu_angular_velocity_[0] = state_msg->imu_state.gyroscope[0];
    hw_imu_angular_velocity_[1] = state_msg->imu_state.gyroscope[1];
    hw_imu_angular_velocity_[2] = state_msg->imu_state.gyroscope[2];
    hw_imu_linear_acceleration_[0] = state_msg->imu_state.accelerometer[0];
    hw_imu_linear_acceleration_[1] = state_msg->imu_state.accelerometer[1];
    hw_imu_linear_acceleration_[2] = state_msg->imu_state.accelerometer[2];

    std::stringstream ss;
    ss << "Read States: [\n";
    for (uint i = 0; i < info_.joints.size(); ++i) {
        ss << std::fixed << std::setprecision(2);
        ss << info_.joints[i].name << ": {" << hw_states_pos_[i] << ", " << hw_states_vel_[i] << ", " << hw_states_eff_[i] << "}";
        if (i < info_.joints.size() - 1) {
            ss << ",\n";
        }
    }
    ss << "\n]";

    // Log the constructed string using RCLCPP_DEBUG_THROTTLE
    RCLCPP_DEBUG_THROTTLE(
        node_->get_logger(), 
        *node_->get_clock(), 
        2000,  // Loggt maximal alle 2000 Millisekunden (2 Sekunden)
        "%s", ss.str().c_str()
    );

    return hardware_interface::return_type::OK;
}

hardware_interface::return_type UnitreeH1HardwareInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/) {
    unitree_hg::msg::LowCmd cmd_msg;

    for (auto& motor : cmd_msg.motor_cmd) {
        motor.mode = 0x01;
        motor.q = 0.0;
        motor.dq = 0.0;
        motor.kp = 0.0;
        motor.kd = 0.0;
        motor.tau = 0.0;
    }
        
    for (uint i = 0; i < info_.joints.size(); i++) {
        int motor_index = urdf_to_actuator_map_[i];
        if (motor_index != -1 && !std::isnan(hw_commands_[i])) {
            cmd_msg.motor_cmd[motor_index].tau = static_cast<float>(hw_commands_[i]);
        }
    }

    // --- Calculate CRC ---
    cmd_msg.mode_pr = HIGHLEVEL;
    cmd_msg.mode_machine = LOWLEVEL;
    cmd_msg.reserve.fill(0);
    get_crc(cmd_msg);

    cmd_pub_->publish(cmd_msg);

    std::stringstream ss;
    ss << "Read Controlls: [\n";
    for (uint i = 0; i < info_.joints.size(); ++i) {
        ss << std::fixed << std::setprecision(2);
        ss << info_.joints[i].name << ": " << hw_commands_[i];
        if (i < info_.joints.size() - 1) {
            ss << ",\n";
        }
    }
    ss << "\n]";

    RCLCPP_DEBUG_THROTTLE(
        node_->get_logger(), 
        *node_->get_clock(), 
        2000,  // Loggt maximal alle 2000 Millisekunden (2 Sekunden)
        "%s", ss.str().c_str()
    );

    return hardware_interface::return_type::OK;
}

void UnitreeH1HardwareInterface::create_urdf_xml_maps() {
    // --- Actuator mapping ---
    urdf_to_actuator_map_.resize(info_.joints.size(), -1);
    std::map<std::string, int> mujoco_actuator_name_to_id;
    for (int i = 0; i < mj_model_->nu; ++i) {
        const char* name = mj_id2name(mj_model_, mjOBJ_ACTUATOR, i);
        if (name) {
            mujoco_actuator_name_to_id[name] = i;
        }
    }
    for (size_t i = 0; i < info_.joints.size(); ++i) {
        if (mujoco_actuator_name_to_id.count(info_.joints[i].name)) {
            urdf_to_actuator_map_[i] = mujoco_actuator_name_to_id[info_.joints[i].name];
        }
    }

    // --- State mapping ---
    urdf_to_sensor_joint_map_.resize(info_.joints.size(), -1);
    std::map<std::string, int> sensor_joint_name_to_id;
    int motor_sensor_idx_counter = 0;
    // Die LowState-Nachricht wird nach der Reihenfolge der 'jointpos'-Sensoren aufgebaut
    for (int i = 0; i < mj_model_->nsensor; ++i) {
        if (mj_model_->sensor_type[i] == mjSENS_JOINTPOS) {
            const char* name = mj_id2name(mj_model_, mjOBJ_SENSOR, i);
            if (name) {
                std::string joint_name = name;
                size_t suffix_pos = joint_name.rfind("_pos");
                if (suffix_pos != std::string::npos) {
                    joint_name.replace(suffix_pos, 4, "_joint");
                }
                sensor_joint_name_to_id[joint_name] = motor_sensor_idx_counter;
                motor_sensor_idx_counter++;
            }
        }
    }
    for (size_t i = 0; i < info_.joints.size(); ++i) {
        if (sensor_joint_name_to_id.count(info_.joints[i].name)) {
            urdf_to_sensor_joint_map_[i] = sensor_joint_name_to_id[info_.joints[i].name];
        }
    }


    // --- DEBUG OUTPUT FOR TESTING ---
    RCLCPP_DEBUG(node_->get_logger(), "===== FINAL JOINT MAPPING =====");
    for (size_t i = 0; i < info_.joints.size(); ++i) {
        RCLCPP_DEBUG(node_->get_logger(), 
            "URDF '%s' (%zu) --> Act [%d], St [%d]",
            info_.joints[i].name.c_str(), i,
            urdf_to_actuator_map_[i],
            urdf_to_sensor_joint_map_[i]
        );
    }
}

}  // namespace unitree_h1_hardware

// Mache diese Klasse als ros2_control Plugin sichtbar
PLUGINLIB_EXPORT_CLASS(
    unitree_h1_hardware::UnitreeH1HardwareInterface,
    hardware_interface::SystemInterface
)