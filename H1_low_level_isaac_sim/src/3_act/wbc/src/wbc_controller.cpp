#include "wbc/wbc_controller.h"
#include "pluginlib/class_list_macros.hpp" // Wichtig für Plugins

namespace wbc
{

WbcController::WbcController() : controller_interface::ControllerInterface() {}

controller_interface::CallbackReturn WbcController::on_init() {
    // Parameter Listener initialisieren
    param_listener_ = std::make_shared<wbc::ParamListener>(get_node());
    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WbcController::on_configure(const rclcpp_lifecycle::State &) {
    RCLCPP_INFO_ONCE(get_node()->get_logger(), "Executing on_configure()");
    
    try {
        param_listener_ = std::make_shared<wbc::ParamListener>(get_node());
        params_ = param_listener_->get_params();
        joint_names_ = params_.joints;

        if (joint_names_.empty()) {
            RCLCPP_ERROR(get_node()->get_logger(), "Der Parameter 'joints' ist nicht definiert oder leer. Controller kann nicht initialisiert werden.");
            return controller_interface::CallbackReturn::ERROR;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Fehler beim Initialisieren der Parameter in on_init(): %s", e.what());
        return controller_interface::CallbackReturn::ERROR;
    }
    
    // Initialisieren Sie den WBC (wie in Ihrem WBCNode Konstruktor)
    wbc::WBCConfig config;
    config.urdf_path = params_.urdf_path;
    config.contact_frame_names = params_.contact_frame_names;
    config.w_com = params_.weights.com;
    config.w_force = params_.weights.force;
    config.w_reg = params_.weights.regularization;
    config.w_contact = params_.weights.contact;
    config.friction_mu = params_.friction_coefficient;
    
    try {
        wbc_ = std::make_unique<wbc::WholeBodyController>(config);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Failed to configure WBC: %s", e.what());
        return controller_interface::CallbackReturn::ERROR;
    }
    int nq = wbc_->get_nq();
    int nv = wbc_->get_nv();
    RCLCPP_INFO(get_node()->get_logger(), "WBC initialized with nq: %d, nv: %d", nq, nv);

    q_current_.resize(nq);
    v_current_.resize(nv);
    q_current_.setZero();
    v_current_.setZero();

    // --- SUBSCRIBER ---
    ref_sub_ = get_node()->create_subscription<wbc_interface::msg::WBCReference>(
        "/robot/desired_state", rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<wbc_interface::msg::WBCReference> msg) {
            rt_ref_buffer_.writeFromNonRT(msg);
        });
    odom_sub_ = get_node()->create_subscription<nav_msgs::msg::Odometry>(
        "/sim/odometry", rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<nav_msgs::msg::Odometry> msg) { 
            rt_odom_buffer_.writeFromNonRT(msg);
        });

    // --- PUBLISHER ---
    auto transient_local_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    foot_poses_pub_ = get_node()->create_publisher<geometry_msgs::msg::PoseArray>(
        "/robot/foot_poses", 
        transient_local_qos
    );

    RCLCPP_INFO(get_node()->get_logger(), "Configuration successful.");
    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration WbcController::command_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (const auto &joint_name : joint_names_) {
        config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_EFFORT);
    }
    return config;
}

controller_interface::InterfaceConfiguration WbcController::state_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (const auto &joint_name : joint_names_) {
        config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
        config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_VELOCITY);
    }
    return config;
}

controller_interface::CallbackReturn WbcController::on_activate(const rclcpp_lifecycle::State &) {
    command_interface_map_.resize(joint_names_.size());
    position_state_interface_map_.resize(joint_names_.size());
    velocity_state_interface_map_.resize(joint_names_.size());

    for (size_t i = 0; i < joint_names_.size(); ++i) {
        const auto& joint_name = joint_names_[i];
        
        // Finde das Command Interface
        const auto command_it = std::find_if(
            command_interfaces_.begin(), command_interfaces_.end(),
            [&joint_name](const auto& interface){ return interface.get_name() == joint_name + "/" + hardware_interface::HW_IF_EFFORT; });
        if (command_it == command_interfaces_.end()) {
            RCLCPP_ERROR(get_node()->get_logger(), "Command interface für Gelenk %s nicht gefunden.", joint_name.c_str());
            return controller_interface::CallbackReturn::ERROR;
        }
        command_interface_map_[i] = std::distance(command_interfaces_.begin(), command_it);

        // Finde das Position State Interface
        const auto pos_it = std::find_if(
            state_interfaces_.begin(), state_interfaces_.end(),
            [&joint_name](const auto& interface){ return interface.get_name() == joint_name + "/" + hardware_interface::HW_IF_POSITION; });
        if (pos_it == state_interfaces_.end()) {
            RCLCPP_ERROR(get_node()->get_logger(), "Position state interface für Gelenk %s nicht gefunden.", joint_name.c_str());
            return controller_interface::CallbackReturn::ERROR;
        }
        position_state_interface_map_[i] = std::distance(state_interfaces_.begin(), pos_it);

        // Finde das Velocity State Interface
        const auto vel_it = std::find_if(
            state_interfaces_.begin(), state_interfaces_.end(),
            [&joint_name](const auto& interface){ return interface.get_name() == joint_name + "/" + hardware_interface::HW_IF_VELOCITY; });
        if (vel_it == state_interfaces_.end()) {
            RCLCPP_ERROR(get_node()->get_logger(), "Velocity state interface für Gelenk %s nicht gefunden.", joint_name.c_str());
            return controller_interface::CallbackReturn::ERROR;
        }
        velocity_state_interface_map_[i] = std::distance(state_interfaces_.begin(), vel_it);
    }

    rt_ref_buffer_.reset();
    rt_odom_buffer_.reset();
    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn WbcController::on_deactivate(const rclcpp_lifecycle::State &) {
    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type WbcController::update(const rclcpp::Time &, const rclcpp::Duration &) {
    RCLCPP_INFO_ONCE(get_node()->get_logger(), "First Time in update.");
    // --- Step 1: Update parameters ---
    this->update_parameters();

    // --- Step 2: read essential sensordata ---
    auto odom_msg = rt_odom_buffer_.readFromRT();
    if (!odom_msg || !*odom_msg) {
        RCLCPP_INFO_ONCE(get_node()->get_logger(), "Warte auf initiale Zustandsdaten (Odom)...");
        return controller_interface::return_type::OK;
    }

    // --- Step 3: build current robot state ---
    read_state_from_hardware(**odom_msg);

    // --- Step 4: prepare reference data ---
    WBCReferenceData refs_local;
    auto wbc_ref_msg_ptr = rt_ref_buffer_.readFromRT();

    if (wbc_ref_msg_ptr && *wbc_ref_msg_ptr) {
        // If WBC reference message is valid, extract the references else hold starting position
        if (!extract_wbc_references(**wbc_ref_msg_ptr, refs_local)) {
            RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, 
                                 "Failure while extracting the WBC-reference. Using 'Holding'-Reference.");
            create_hold_position_reference(refs_local);
        }
    } else {
        RCLCPP_INFO_ONCE(get_node()->get_logger(), "No WBC-reference received yet. Using 'Holding'-Reference.");
        create_hold_position_reference(refs_local);
    }

    // --- Step 5: Solve WBC ---
    if (wbc_->solve(q_current_, v_current_, refs_local)) {
        const auto& torques = wbc_->get_joint_torques();
        if (!write_torques_to_hardware(torques)) {
             return controller_interface::return_type::ERROR;
        }
        publish_foot_poses();

    } else {
        RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "WBC solver failed!");
        const Eigen::VectorXd zero_torques = Eigen::VectorXd::Zero(command_interfaces_.size());
        write_torques_to_hardware(zero_torques);
    }

    return controller_interface::return_type::OK;
}

void WbcController::update_parameters() {
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        wbc::WBCConfig config;
        // ... (Code zum Füllen der config-Struktur) ...
        config.w_com = params_.weights.com;
        config.w_force = params_.weights.force;
        config.w_reg = params_.weights.regularization;
        config.w_contact = params_.weights.contact;
        config.friction_mu = params_.friction_coefficient;
        wbc_->update_config(config);
        RCLCPP_INFO(get_node()->get_logger(), "WBC-Parameter updated.");
    }
}

bool WbcController::extract_wbc_references(const wbc_interface::msg::WBCReference& msg, wbc::WBCReferenceData& refs_local) {
    if (static_cast<size_t>(refs_local.f_c_des.size()) != msg.f_c_des.size()) {
        RCLCPP_DEBUG_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), 1000,
            "Size of the contact force vector does not match! Expected: %ld, Got: %zu",
            refs_local.f_c_des.size(), msg.f_c_des.size());
        return false;
    }

    refs_local.f_c_des.setZero(3 * params_.contact_frame_names.size());
    refs_local.a_com_des << msg.a_com_des.x, msg.a_com_des.y, msg.a_com_des.z;
    refs_local.a_ang_des << msg.a_ang_des.x, msg.a_ang_des.y, msg.a_ang_des.z;
    refs_local.f_c_des = Eigen::Map<const Eigen::VectorXd>(msg.f_c_des.data(), msg.f_c_des.size());

    return true;
}

void WbcController::read_state_from_hardware(const nav_msgs::msg::Odometry& odom_data) {
    RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "Reading state from hardware...");

    // Floating Base aus Odometrie
    q_current_(0) = odom_data.pose.pose.position.x;
    q_current_(1) = odom_data.pose.pose.position.y;
    q_current_(2) = odom_data.pose.pose.position.z;
    q_current_(3) = odom_data.pose.pose.orientation.x;
    q_current_(4) = odom_data.pose.pose.orientation.y;
    q_current_(5) = odom_data.pose.pose.orientation.z;
    q_current_(6) = odom_data.pose.pose.orientation.w;

    v_current_(0) = odom_data.twist.twist.linear.x;
    v_current_(1) = odom_data.twist.twist.linear.y;
    v_current_(2) = odom_data.twist.twist.linear.z;
    v_current_(3) = odom_data.twist.twist.angular.x;
    v_current_(4) = odom_data.twist.twist.angular.y;
    v_current_(5) = odom_data.twist.twist.angular.z;

    // Gelenkzustände über die robuste Zuordnung lesen
    for (size_t i = 0; i < joint_names_.size(); ++i) {
        q_current_(7 + i) = state_interfaces_[position_state_interface_map_[i]].get_value();
        v_current_(6 + i) = state_interfaces_[velocity_state_interface_map_[i]].get_value();
    }
}

bool WbcController::write_torques_to_hardware(const Eigen::VectorXd& torques) {
    if (static_cast<size_t>(torques.size()) != command_interfaces_.size()) {
        RCLCPP_ERROR(get_node()->get_logger(), "Size of the torque vector (%ld) does not match the number of command interfaces (%zu)!", torques.size(), command_interfaces_.size());
        return false;
    }

    std::stringstream ss;
    ss << "Sending Torques: ";
    for (size_t i = 0; i < command_interfaces_.size(); ++i) {
        const auto& interface_name = command_interfaces_[command_interface_map_[i]].get_name();
        double torque_value = torques(i);
        command_interfaces_[command_interface_map_[i]].set_value(torque_value);
        ss << interface_name << ": " << std::fixed << std::setprecision(2) << torque_value << " | ";
    }
    RCLCPP_DEBUG_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, ss.str().c_str());
    return true;
}

void WbcController::publish_foot_poses() {
    auto poses_msg = std::make_unique<geometry_msgs::msg::PoseArray>();
    poses_msg->header.stamp = get_node()->get_clock()->now();
    poses_msg->header.frame_id = "odom"; // Positionen im Welt-Frame

    for (const auto& frame_name : params_.contact_frame_names) {
        try {
            pinocchio::SE3 frame_placement = wbc_->get_frame_placement(frame_name);
            geometry_msgs::msg::Pose pose;

            pose.position.x = frame_placement.translation().x();
            pose.position.y = frame_placement.translation().y();
            pose.position.z = frame_placement.translation().z();
            
            Eigen::Quaterniond q(frame_placement.rotation());
            pose.orientation.x = q.x();
            pose.orientation.y = q.y();
            pose.orientation.z = q.z();
            pose.orientation.w = q.w();

            poses_msg->poses.push_back(pose);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_node()->get_logger(), "Failure while retrieving frame placement for %s: %s", frame_name.c_str(), e.what());
        }
    }
    foot_poses_pub_->publish(std::move(poses_msg));
}

void WbcController::create_hold_position_reference(wbc::WBCReferenceData& refs) const {
    // Target - No movement.
    refs.a_com_des.setZero();
    refs.a_ang_des.setZero();

    double mass = wbc_->get_total_mass();
    double gravity = wbc_->get_z_gravity();
    double total_weight = mass * gravity;
    double force_per_foot = total_weight / params_.contact_frame_names.size();

    refs.f_c_des.setZero(3 * params_.contact_frame_names.size());
    for (size_t i = 0; i < params_.contact_frame_names.size(); ++i) {
        refs.f_c_des(3 * i + 2) = force_per_foot; // Setze nur die vertikale Kraft (Fz)
    }
}

} // namespace wbc

PLUGINLIB_EXPORT_CLASS(wbc::WbcController, controller_interface::ControllerInterface)