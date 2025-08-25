#include "mpc_node.h"

// --- Constructor & Destructor ---

AcadosMPCNode::AcadosMPCNode()
    : Node("mpc_node"), ocp_capsule_(nullptr),
      state_received_(false), yref_received_(false), parameter_received_(false) 
{
    RCLCPP_INFO(this->get_logger(), "Initializing SRBM Acados OCP MPC Node...");

    // 1. Parameter Handling
    this->declare_parameters();
    this->load_parameters();
    this->config_.mass = 34.0;
    this->config_.gravity = 9.81;
    this->log_parameters();
    param_callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&AcadosMPCNode::on_parameter_update, this, std::placeholders::_1)
    );

    // 2. Initialize Acados Solver
    this->initialize_ocp_solver();
    
    // 3. Setup ROS Communications
    this->state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimator/odometry", 10,
        std::bind(&AcadosMPCNode::state_callback, this, std::placeholders::_1)
    );
    this->reference_sub_ = this->create_subscription<mpc_interface::msg::YReference>(
        "/robot/reference", 10,
        std::bind(&AcadosMPCNode::reference_callback, this, std::placeholders::_1)
    );
    this->foot_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
        "/robot/foot_poses", 10,
        std::bind(&AcadosMPCNode::parameter_callback, this, std::placeholders::_1)
    );
    this->wbc_ref_pub_ = this->create_publisher<wbc_interface::msg::WBCReference>("/robot/desired_state", 10);
    // this->viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/mpc/viz", 10);

    this->timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10), // Anpassen der Timer-Periode!
        std::bind(&AcadosMPCNode::control_loop_callback, this));
}

AcadosMPCNode::~AcadosMPCNode() {
    RCLCPP_INFO(this->get_logger(), "Shutting down and freeing Acados solver memory.");
    if (this->ocp_capsule_) {
        int status = srbm_robot_acados_free(this->ocp_capsule_);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "srbm_robot_acados_free() returned status %d.", status);
        }
        status = srbm_robot_acados_free_capsule(this->ocp_capsule_);
        if (status) {
            RCLCPP_ERROR(this->get_logger(), "srbm_robot_acados_free_capsule() returned status %d.", status);
        }
    }
}

// --- ROS Callbacks ---
void AcadosMPCNode::state_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::scoped_lock lock(data_mutex_); 
    this->msg2state(this->current_state_, msg);
    if (!this->state_received_) this->state_received_ = true;
}

void AcadosMPCNode::reference_callback(const mpc_interface::msg::YReference::SharedPtr msg) {
    std::scoped_lock lock(data_mutex_); 
    this->msg2yref(this->yref_, this->yref_e_, msg);
    if (!this->yref_received_) this->yref_received_ = true;
}

void AcadosMPCNode::parameter_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    std::scoped_lock lock(data_mutex_); 
    this->msg2param(this->parameter_, msg);
    if (!this->parameter_received_) this->parameter_received_ = true;
}

void AcadosMPCNode::control_loop_callback() {
    if (!(this->state_received_ && this->yref_received_ && this->parameter_received_)){
        RCLCPP_INFO_STREAM_ONCE(this->get_logger(), "Wait on state-, reference- and parameter-message...");
        return;
    }

    double x0_local[SRBM_ROBOT_NBX0];
    double yref_local[SRBM_ROBOT_N][SRBM_ROBOT_NY];
    double yref_e_local[SRBM_ROBOT_NYN];
    double p_local[SRBM_ROBOT_NP];
    {
        std::scoped_lock lock(data_mutex_);
        std::copy(std::begin(this->current_state_), std::end(this->current_state_), std::begin(x0_local));
        std::copy(&this->yref_[0][0], &this->yref_[0][0] + SRBM_ROBOT_N*SRBM_ROBOT_NY, &yref_local[0][0]);
        std::copy(std::begin(this->yref_e_), std::end(this->yref_e_), std::begin(yref_e_local));
        std::copy(std::begin(this->parameter_), std::end(this->parameter_), std::begin(p_local));
    }
    
    // OCP mit den lokalen Kopien lösen
    this->update_x0(x0_local);
    this->update_yref(yref_local);
    this->update_yref_e(yref_e_local);
    this->update_ocp_parameters(p_local);

    int status = srbm_robot_acados_solve(ocp_capsule_);
    if (status != ACADOS_SUCCESS) {
        RCLCPP_ERROR(this->get_logger(), "Solver failed with status %d.", status);
    } else {
        this->publish_solution();
    }
}

// --- Acados Solver Interaction ---
void AcadosMPCNode::initialize_ocp_solver() {
    this->ocp_capsule_ = srbm_robot_acados_create_capsule();
    int status = srbm_robot_acados_create(ocp_capsule_);
    if (status) {
        RCLCPP_FATAL(this->get_logger(), "acados_create() failed with status %d. Shutting down.", status);
        rclcpp::shutdown();
        return;
    }

    this->ocp_nlp_config_ = srbm_robot_acados_get_nlp_config(this->ocp_capsule_);
    this->ocp_nlp_dims_ = srbm_robot_acados_get_nlp_dims(this->ocp_capsule_);
    this->ocp_nlp_in_ = srbm_robot_acados_get_nlp_in(this->ocp_capsule_);
    this->ocp_nlp_out_ = srbm_robot_acados_get_nlp_out(this->ocp_capsule_);
    this->ocp_nlp_opts_ = srbm_robot_acados_get_nlp_opts(this->ocp_capsule_);

    // Set initial weights and constraints from parameters
    update_solver_weights();
    update_solver_constraints();
    RCLCPP_INFO(this->get_logger(), "Acados OCP Solver initialized successfully.");
}

void AcadosMPCNode::update_x0(double x0[SRBM_ROBOT_NBX0]) {
    ocp_nlp_constraints_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, ocp_nlp_out_, 0, "lbx", x0);
    ocp_nlp_constraints_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, ocp_nlp_out_, 0, "ubx", x0);
}

void AcadosMPCNode::update_yref(double yref[SRBM_ROBOT_N][SRBM_ROBOT_NY]) {
    for (int i = 0; i < SRBM_ROBOT_N; i++) {
        ocp_nlp_cost_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, i, "yref", yref);
    }
}

void AcadosMPCNode::update_yref_e(double yref_e[SRBM_ROBOT_NYN]) {
    ocp_nlp_cost_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, SRBM_ROBOT_N, "yref", yref_e);
}

void AcadosMPCNode::update_ocp_parameters(double p[SRBM_ROBOT_NP]) {
    for (int k = 0; k < SRBM_ROBOT_N; ++k) {
        srbm_robot_acados_update_params(this->ocp_capsule_, k, p, SRBM_ROBOT_NP);
    }
}

void AcadosMPCNode::update_solver_weights() {
    // Cost is LINEAR_LS: 0.5 * || Vx*x + Vu*u - y_ref ||^2_W
    // W = block_diag(Q, R)
    // Stage Weight
    Eigen::Matrix<double, SRBM_ROBOT_NY, SRBM_ROBOT_NY> W = Eigen::Matrix<double, SRBM_ROBOT_NY, SRBM_ROBOT_NY>::Zero();
    for(int i=0; i<SRBM_ROBOT_NX; ++i) W(i,i) = this->config_.w_q[i];
    for(int i=0; i<SRBM_ROBOT_NU; ++i) W(SRBM_ROBOT_NX + i, SRBM_ROBOT_NX + i) = this->config_.w_r[i];
    
    // Set cost for all intermediate stages
    for (int i = 0; i < SRBM_ROBOT_N; i++) { 
        ocp_nlp_cost_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, i, "W", W.data());
    }
    
    // Terminal Weight
    Eigen::Matrix<double, SRBM_ROBOT_NX, SRBM_ROBOT_NX> W_e = Eigen::Matrix<double, SRBM_ROBOT_NX, SRBM_ROBOT_NX>::Zero();
    for(int i=0; i<SRBM_ROBOT_NX; ++i) W_e(i,i) = this->config_.w_q[i]; // Using same Q weights for terminal cost
    
    ocp_nlp_cost_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, SRBM_ROBOT_N, "W", W_e.data());
}

void AcadosMPCNode::update_solver_constraints() {
    double f_max = this->config_.force_max;
    double lbu[SRBM_ROBOT_NU] = {-f_max, -f_max, 0.0, -f_max, -f_max, 0.0};
    double ubu[SRBM_ROBOT_NU] = { f_max,  f_max, f_max,  f_max,  f_max, f_max};
    
    for (int i=0; i < SRBM_ROBOT_N; i++) {
        ocp_nlp_constraints_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, ocp_nlp_out_, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_in_, ocp_nlp_out_, i, "ubu", ubu);
    }
}

void AcadosMPCNode::publish_solution() {
    double u0[SRBM_ROBOT_NU];
    double x0[SRBM_ROBOT_NX];
    double x1[SRBM_ROBOT_NX];

    this->get_input_at_stage(u0, 0); // u0 = [fx1, fy1, fz1, fx2, fy2, fz2]
    this->get_state_at_stage(x0, 0);
    this->get_state_at_stage(x1, 1);

    // WBC-Reference-Message
    auto wbc_ref_msg = std::make_unique<wbc_interface::msg::WBCReference>();
    wbc_ref_msg->header.stamp = this->get_clock()->now();
    wbc_ref_msg->header.frame_id = "wbc_ref";

    double dt = 0.02;
    // double tf; // TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // ocp_nlp_opts_get(this->ocp_nlp_config_, this->ocp_nlp_opts_, "tf", &tf);
    // double dt = tf / static_cast<double>(SRBM_ROBOT_N);

    wbc_ref_msg->a_com_des.x = (x1[7] - x0[7]) / dt;
    wbc_ref_msg->a_com_des.y = (x1[8] - x0[8]) / dt;
    wbc_ref_msg->a_com_des.z = (x1[9] - x0[9]) / dt;

    // Angulare Beschleunigung als Differenzenquotient: alpha = (w1 - w0) / dt
    wbc_ref_msg->a_ang_des.x = (x1[10] - x0[10]) / dt;
    wbc_ref_msg->a_ang_des.y = (x1[11] - x0[11]) / dt;
    wbc_ref_msg->a_ang_des.z = (x1[12] - x0[12]) / dt;

    wbc_ref_msg->f_c_des.assign(u0, u0 + SRBM_ROBOT_NU);

    this->wbc_ref_pub_->publish(std::move(wbc_ref_msg));
}

// --- Parameter Handling Implementation ---

void AcadosMPCNode::declare_parameters() {
    this->declare_parameter("mpc.constraints.force_max", 100.0);
    this->declare_parameter("mpc.weights.q", std::vector<double>(13, 0.0));
    this->declare_parameter("mpc.weights.r", std::vector<double>(6, 0.0));
}

void AcadosMPCNode::load_parameters() {
    this->get_parameter("mpc.constraints.force_max", this->config_.force_max);
    this->get_parameter("mpc.weights.q", this->config_.w_q);
    this->get_parameter("mpc.weights.r", this->config_.w_r);
}

void AcadosMPCNode::log_parameters() {
    RCLCPP_INFO(this->get_logger(), "--- SRBM MPC Configuration ---");
    RCLCPP_INFO(this->get_logger(), "Constraints: force_max=%.2f", this->config_.force_max);

    // Log weight vectors q and r
    std::ostringstream q_stream, r_stream;
    q_stream << "[";
    for (size_t i = 0; i < this->config_.w_q.size(); ++i) {
        q_stream << this->config_.w_q[i];
        if (i != this->config_.w_q.size() - 1) q_stream << ", ";
    }
    q_stream << "]";
    r_stream << "[";
    for (size_t i = 0; i < this->config_.w_r.size(); ++i) {
        r_stream << this->config_.w_r[i];
        if (i != this->config_.w_r.size() - 1) r_stream << ", ";
    }
    r_stream << "]";
    RCLCPP_INFO(this->get_logger(), "Weights: q=%s, r=%s", q_stream.str().c_str(), r_stream.str().c_str());
    RCLCPP_INFO(this->get_logger(), "--------------------------------");
}

rcl_interfaces::msg::SetParametersResult AcadosMPCNode::on_parameter_update(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    load_parameters(); 

    for (const auto& param : params) {
        if (param.get_name() == "mpc.weights.w_r" || param.get_name() == "mpc.weights.w_q") {
            update_solver_weights();
        } else if (param.get_name() == "mpc.constraints.force_max") {
            update_solver_constraints();
        } 
    }

    RCLCPP_INFO(this->get_logger(), "Parameters updated successfully.");
    log_parameters();
    return result;
}

// --- Getter & Utility Methods ---
void AcadosMPCNode::get_input_at_stage(double u[SRBM_ROBOT_NU], int stage) {
    ocp_nlp_out_get(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_out_, stage, "u", u);
}

void AcadosMPCNode::get_state_at_stage(double x[SRBM_ROBOT_NX], int stage) {
    ocp_nlp_out_get(ocp_nlp_config_, ocp_nlp_dims_, ocp_nlp_out_, stage, "x", x);
}

void AcadosMPCNode::msg2state(double x0[SRBM_ROBOT_NX], const nav_msgs::msg::Odometry::SharedPtr msg) {
    x0[0] = msg->pose.pose.position.x;
    x0[1] = msg->pose.pose.position.y;
    x0[2] = msg->pose.pose.position.z;
    x0[3] = msg->pose.pose.orientation.w;
    x0[4] = msg->pose.pose.orientation.x;
    x0[5] = msg->pose.pose.orientation.y;
    x0[6] = msg->pose.pose.orientation.z;
    x0[7] = msg->twist.twist.linear.x;
    x0[8] = msg->twist.twist.linear.y;
    x0[9] = msg->twist.twist.linear.z;
    x0[10] = msg->twist.twist.angular.x;
    x0[11] = msg->twist.twist.angular.y;
    x0[12] = msg->twist.twist.angular.z;
}

void AcadosMPCNode::msg2param(double param[SRBM_ROBOT_NP], const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    param[0] = msg->poses[0].position.x;
    param[1] = msg->poses[0].position.y;
    param[2] = msg->poses[0].position.z;
    param[3] = msg->poses[1].position.x;
    param[4] = msg->poses[1].position.y;
    param[5] = msg->poses[1].position.z;
}

void AcadosMPCNode::msg2yref(
        double yref[SRBM_ROBOT_N][SRBM_ROBOT_NY], 
        double yref_e[SRBM_ROBOT_NYN], 
        const mpc_interface::msg::YReference::SharedPtr msg
) {
    if (msg->poses.size() != SRBM_ROBOT_N + 1 ||
        msg->twists.size() != SRBM_ROBOT_N + 1 ||
        msg->f_c1.size() != SRBM_ROBOT_N ||
        msg->f_c2.size() != SRBM_ROBOT_N)
    {
        RCLCPP_ERROR(this->get_logger(),
                     "YReference message has incorrect array sizes. Expected %d poses/twists and %d forces, but got %zu poses, %zu twists, %zu f_c1, %zu f_c2.",
                     SRBM_ROBOT_N + 1, SRBM_ROBOT_N,
                     msg->poses.size(), msg->twists.size(), msg->f_c1.size(), msg->f_c2.size());
        return; // Nicht weitermachen, da die Daten ungültig sind.
    }

    // Stage Y-reference
    for (int i = 0; i < SRBM_ROBOT_N; ++i) {
        const auto& current_pos = msg->poses[i];
        const auto& current_twist = msg->twists[i];
        const auto& current_fc1 = msg->f_c1[i];
        const auto& current_fc2 = msg->f_c2[i];

        // STATES
        yref[i][0] = current_pos.position.x;
        yref[i][1] = current_pos.position.y;
        yref[i][2] = current_pos.position.z;
        yref[i][3] = current_pos.orientation.w;
        yref[i][4] = current_pos.orientation.x;
        yref[i][5] = current_pos.orientation.y;
        yref[i][6] = current_pos.orientation.z;
        yref[i][7] = current_twist.linear.x;
        yref[i][8] = current_twist.linear.y;
        yref[i][9] = current_twist.linear.z;
        yref[i][10] = current_twist.angular.x;
        yref[i][11] = current_twist.angular.y;
        yref[i][12] = current_twist.angular.z;

        // INPUTS
        yref[i][13] = current_fc1.x;
        yref[i][14] = current_fc1.y;
        yref[i][15] = current_fc1.z;
        yref[i][16] = current_fc2.x;
        yref[i][17] = current_fc2.y;
        yref[i][18] = current_fc2.z;
    }

    // Terminal Y-reference
    const auto& current_pos = msg->poses[SRBM_ROBOT_N];
    const auto& current_twist = msg->twists[SRBM_ROBOT_N];

    yref_e[0] = current_pos.position.x;
    yref_e[1] = current_pos.position.y;
    yref_e[2] = current_pos.position.z;
    yref_e[3] = current_pos.orientation.w;
    yref_e[4] = current_pos.orientation.x;
    yref_e[5] = current_pos.orientation.y;
    yref_e[6] = current_pos.orientation.z;
    yref_e[7] = current_twist.linear.x;
    yref_e[8] = current_twist.linear.y;
    yref_e[9] = current_twist.linear.z;
    yref_e[10] = current_twist.angular.x;
    yref_e[11] = current_twist.angular.y;
    yref_e[12] = current_twist.angular.z;
}

// --- Main Function ---
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AcadosMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}