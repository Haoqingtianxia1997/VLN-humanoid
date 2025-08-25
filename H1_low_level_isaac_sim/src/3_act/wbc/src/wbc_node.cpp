#include "wbc_node.h"

WBCNode::WBCNode() : Node("wbc_node") 
{
    this->config_ = WBCConfig();

    // 1. Handle ROS Parameters
    this->declare_parameters();
    this->load_parameters();
    this->log_parameters();
    this->param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&WBCNode::on_parameter_update, this, std::placeholders::_1));

    // 2. Initialize the Whole Body Controller
    try {
        this->wbc_ = std::make_unique<WholeBodyController>(this->config_);
        // Apply friction parameter from config
    } catch (const std::runtime_error& e) {
        RCLCPP_FATAL(this->get_logger(), "Failed to initialize WBC: %s. Shutting down.", e.what());
        rclcpp::shutdown();
        return;
    }
    
    // Resize state vectors based on the model
    q_current_.resize(wbc_->get_nq());
    v_current_.resize(wbc_->get_nv());
    mpc_refs_.f_c_des.resize(3 * wbc_->get_num_contacts());


    // 3. Setup ROS Communications
    this->joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10, std::bind(&WBCNode::joint_state_callback, this, std::placeholders::_1));

    this->mpc_ref_sub_ = this->create_subscription<wbc_interface::msg::WBCReference>(
        "wbc_references", 10, std::bind(&WBCNode::mpc_ref_callback, this, std::placeholders::_1));

    this->torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("wbc_torques", 10);

    // 4. Start the main control loop timer
    this->timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / this->update_rate_hz_),
        std::bind(&WBCNode::control_loop_callback, this));

    RCLCPP_INFO(this->get_logger(), "WBC Node has been initialized successfully.");
}

WBCNode::~WBCNode() 
{
}

void WBCNode::control_loop_callback() 
{
    if (!this->has_state_ || !this->has_refs_) {
        RCLCPP_WARN_ONCE(this->get_logger(), "Waiting for initial robot state and MPC references...");
        return;
    }

    // Thread-safe copy of state and reference data
    Eigen::VectorXd q_local, v_local;
    WBCReferenceData refs_local;
    {
        std::scoped_lock lock(this->state_mutex_, this->ref_mutex_);
        q_local = this->q_current_;
        v_local = this->v_current_;
        refs_local = this->mpc_refs_;
    }

    // Solve the WBC QP
    if (this->wbc_->solve(q_local, v_local, refs_local)) {
        // Get and publish the resulting torques
        auto torques = this->wbc_->get_joint_torques();
        std_msgs::msg::Float64MultiArray torque_msg;
        torque_msg.data.assign(torques.data(), torques.data() + torques.size());
        this->torque_pub_->publish(torque_msg);
    } else {
        RCLCPP_ERROR(this->get_logger(), "WBC solver failed!");
    }
}

void WBCNode::joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->state_mutex_);
    // IMPORTANT: Ensure the order of joints in the message matches Pinocchio's model.
    // This example assumes a floating base, so the first 7 elements are for the base.
    // You may need a more sophisticated mapping from message names to vector indices.
    if (msg->position.size() == static_cast<size_t>(this->wbc_->get_nq() - 6) && msg->velocity.size() == static_cast<size_t>(this->wbc_->get_nv() - 6)) {
        // Assuming a floating base (e.g., humanoid or quadruped)
        // Base state would typically come from odometry or a state estimator.
        // Here we just fill the actuated joint part.
        // For now, let's assume base is at identity. THIS IS A PLACEHOLDER.
        this->q_current_.head<7>().setZero();
        this->q_current_[6] = 1.0; // Quaternion w = 1
        this->v_current_.head<6>().setZero();
        
        // Fill actuated joint values
        for(size_t i = 0; i < msg->position.size(); ++i) {
            this->q_current_[6 + i] = msg->position[i];
            this->v_current_[6 + i] = msg->velocity[i];
        }
        this->has_state_ = true;
    } else {
        RCLCPP_WARN_ONCE(this->get_logger(), "JointState message size does not match model dimensions. Expected %zu pos, %zu vel.", static_cast<size_t>(this->wbc_->get_nq() - 6), static_cast<size_t>(this->wbc_->get_nv() - 6));
    }
}

void WBCNode::mpc_ref_callback(const wbc_interface::msg::WBCReference::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->ref_mutex_);
    this->mpc_refs_.a_com_des << msg->a_com_des.x, msg->a_com_des.y, msg->a_com_des.z;
    this->mpc_refs_.a_ang_des << msg->a_ang_des.x, msg->a_ang_des.y, msg->a_ang_des.z;

    if (msg->f_c_des.size() == static_cast<size_t>(this->mpc_refs_.f_c_des.size())) {
        this->mpc_refs_.f_c_des = Eigen::Map<const Eigen::VectorXd>(msg->f_c_des.data(), msg->f_c_des.size());
        this->has_refs_ = true;
    } else {
        RCLCPP_WARN(this->get_logger(), "Desired contact force vector size mismatch. Expected %ld, got %ld.", this->mpc_refs_.f_c_des.size(), msg->f_c_des.size());
    }
}

// --- Parameter Handling ---

void WBCNode::declare_parameters() {
    this->declare_parameter("urdf_path", std::string(""));
    this->declare_parameter("contact_frame_names", std::vector<std::string>({}));
    this->declare_parameter("update_rate_hz", 100.0);
    this->declare_parameter("wbc.weights.com", 1.0);
    this->declare_parameter("wbc.weights.force", 1e-3);
    this->declare_parameter("wbc.weights.regularization", 1e-4);
    this->declare_parameter("wbc.friction_coefficient", 0.7);
}

void WBCNode::load_parameters() {
    this->get_parameter("update_rate_hz", this->update_rate_hz_);
    this->get_parameter("urdf_path", this->config_.urdf_path);
    this->get_parameter("contact_frame_names", this->config_.contact_frame_names);
    this->get_parameter("wbc.weights.com", this->config_.w_com);
    this->get_parameter("wbc.weights.force", this->config_.w_force);
    this->get_parameter("wbc.weights.regularization", this->config_.w_reg);
    this->get_parameter("wbc.friction_coefficient", this->config_.friction_mu);
}

void WBCNode::log_parameters() {
    RCLCPP_INFO(this->get_logger(), "--- WBC Node Configuration ---");
    RCLCPP_INFO(this->get_logger(), "Update Rate (Hz): %.1f", this->update_rate_hz_);
    RCLCPP_INFO(this->get_logger(), "URDF Path: %s", this->config_.urdf_path.c_str());
    std::string frames_str = "";
    for(const auto& name : this->config_.contact_frame_names) frames_str += name + ", ";
    RCLCPP_INFO(this->get_logger(), "Contact Frames: [%s]", frames_str.c_str());
    RCLCPP_INFO(this->get_logger(), "CoM Weight: %f", this->config_.w_com);
    RCLCPP_INFO(this->get_logger(), "Force Weight: %f", this->config_.w_force);
    RCLCPP_INFO(this->get_logger(), "Regularization Weight: %f", this->config_.w_reg);
    RCLCPP_INFO(this->get_logger(), "Friction Coefficient: %f", this->config_.friction_mu);
    RCLCPP_INFO(this->get_logger(), "--------------------------------");
}

rcl_interfaces::msg::SetParametersResult WBCNode::on_parameter_update(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true; // Assume success unless something fails

    for (const auto& param : params) {
        // Check for parameters that require a restart
        if (param.get_name() == "urdf_path" || param.get_name() == "contact_frame_names") {
            RCLCPP_WARN(this->get_logger(), "Changing '%s' requires a node restart to take effect.", param.get_name().c_str());
            result.successful = false;
            result.reason = "Changing the model structure requires a restart.";
            return result;
        }

        // Update the specific parameter that changed
        if (param.get_name() == "wbc.weights.com") {
            this->config_.w_com = param.as_double();
        } else if (param.get_name() == "wbc.weights.force") {
            this->config_.w_force = param.as_double();
        } else if (param.get_name() == "wbc.weights.regularization") {
            this->config_.w_reg = param.as_double();
        } else if (param.get_name() == "wbc.friction_coefficient") {
            this->config_.friction_mu = param.as_double();
        }
        else if (param.get_name() == "update_rate_hz") {
            this->update_rate_hz_ = param.as_double();
            // If this controls a timer, you might need to reset the timer here
        }
    }

    // After successfully processing all changes, log the new state
    if (result.successful) {
        RCLCPP_INFO(this->get_logger(), "Parameters updated successfully. New configuration:");
        log_parameters();
    }
    return result;
}


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto wbc_node = std::make_shared<WBCNode>();
    rclcpp::spin(wbc_node);
    rclcpp::shutdown();
    return 0;
}