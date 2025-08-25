#include "reference_generator_node.h"

ReferenceGeneratorNode::ReferenceGeneratorNode() : Node("reference_generator_node") {
    // Subscribe to high-level commands (e.g., desired velocity)
    cmd_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
        "/target/pose", 10,
        std::bind(&ReferenceGeneratorNode::pose_callback, this, std::placeholders::_1)
    );
    cmd_twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/target/twist", 10,
        std::bind(&ReferenceGeneratorNode::twist_callback, this, std::placeholders::_1)
    );

    // Publish the YReference message
    yref_pub_ = this->create_publisher<mpc_interface::msg::YReference>(
        "/robot/reference", 10);
}

void ReferenceGeneratorNode::pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    std::array<double, SRBM_ROBOT_NX> init_state = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    // auto result = slow_walk(init_state, 0);
    // this->publish_yref(yrefs, yref_e);
}

void ReferenceGeneratorNode::twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::array<double, SRBM_ROBOT_NX> init_state = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    auto [yrefs, yref_e] = slow_walk(init_state, msg->linear.x, msg->linear.y, msg->linear.z, msg->angular.y, 0.8, 0.5);
    this->publish_yref(yrefs, yref_e);
}

void ReferenceGeneratorNode::publish_yref(
        std::vector<std::array<double, SRBM_ROBOT_NY>> yrefs, 
        std::array<double, SRBM_ROBOT_NYN> yref_e
) {
    RCLCPP_INFO_ONCE(this->get_logger(), "Publishing YReference message...");
    auto yref_msg = std::make_unique<mpc_interface::msg::YReference>();
    yref_msg->header.stamp = this->get_clock()->now();

    yref_msg->poses.reserve(SRBM_ROBOT_N + 1);
    yref_msg->twists.reserve(SRBM_ROBOT_N + 1);
    yref_msg->f_c1.reserve(SRBM_ROBOT_N);
    yref_msg->f_c2.reserve(SRBM_ROBOT_N);

    // --- Populate for N stages (0 to SRBM_ROBOT_N-1) ---
    for (int i = 0; i < SRBM_ROBOT_N; ++i) {
        geometry_msgs::msg::Pose current_pose;
        geometry_msgs::msg::Twist current_twist;
        geometry_msgs::msg::Vector3 current_fc1;
        geometry_msgs::msg::Vector3 current_fc2;

        // Pose (Indices 0-6)
        current_pose.position.x = yrefs[i][0];
        current_pose.position.y = yrefs[i][1];
        current_pose.position.z = yrefs[i][2];
        current_pose.orientation.w = yrefs[i][3];
        current_pose.orientation.x = yrefs[i][4];
        current_pose.orientation.y = yrefs[i][5];
        current_pose.orientation.z = yrefs[i][6];
        yref_msg->poses.push_back(current_pose);

        // Twist (Indices 7-12)
        current_twist.linear.x = yrefs[i][7];
        current_twist.linear.y = yrefs[i][8];
        current_twist.linear.z = yrefs[i][9];
        current_twist.angular.x = yrefs[i][10];
        current_twist.angular.y = yrefs[i][11];
        current_twist.angular.z = yrefs[i][12];
        yref_msg->twists.push_back(current_twist);

        // f_c1 (Indices 13-15)
        current_fc1.x = yrefs[i][13];
        current_fc1.y = yrefs[i][14];
        current_fc1.z = yrefs[i][15];
        yref_msg->f_c1.push_back(current_fc1);

        // f_c2 (Indices 16-18)
        current_fc2.x = yrefs[i][16];
        current_fc2.y = yrefs[i][17];
        current_fc2.z = yrefs[i][18];
        yref_msg->f_c2.push_back(current_fc2);
    }

    // --- Populate for Terminal Stage (SRBM_ROBOT_N) using yref_e ---
    geometry_msgs::msg::Pose terminal_pose;
    geometry_msgs::msg::Twist terminal_twist;

    // Pose (Indices 0-6 from yref_e)
    terminal_pose.position.x = yref_e[0];
    terminal_pose.position.y = yref_e[1];
    terminal_pose.position.z = yref_e[2];
    terminal_pose.orientation.w = yref_e[3];
    terminal_pose.orientation.x = yref_e[4];
    terminal_pose.orientation.y = yref_e[5];
    terminal_pose.orientation.z = yref_e[6];
    yref_msg->poses.push_back(terminal_pose);

    // Twist (Indices 7-12 from yref_e)
    terminal_twist.linear.x = yref_e[7];
    terminal_twist.linear.y = yref_e[8];
    terminal_twist.linear.z = yref_e[9];
    terminal_twist.angular.x = yref_e[10];
    terminal_twist.angular.y = yref_e[11];
    terminal_twist.angular.z = yref_e[12];
    yref_msg->twists.push_back(terminal_twist);

    yref_pub_->publish(std::move(yref_msg));
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReferenceGeneratorNode>());
    rclcpp::shutdown();
    return 0;
}