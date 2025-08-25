#include "sim_odom_publisher/sim_odom_publisher_node.hpp"

SimOdomPublisher::SimOdomPublisher() : Node("sim_odom_publisher_node") {
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/sim/odometry", 10);
    sport_mode_state_sub_ = this->create_subscription<unitree_go::msg::SportModeState>(
        "/sportmodestate", 10,
        std::bind(&SimOdomPublisher::sportModeCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "Sim Odometry Publisher Node started.");
}

void SimOdomPublisher::sportModeCallback(const unitree_go::msg::SportModeState::SharedPtr msg) const {
    auto odom_msg = std::make_unique<nav_msgs::msg::Odometry>();

    // odom_msg->header.stamp = msg->stamp;
    odom_msg->header.frame_id = "odom";
    odom_msg->child_frame_id = "pelvis"; // Passe dies an deinen base_link an

    // Fülle Position und Orientierung
    odom_msg->pose.pose.position.x = msg->position[0];
    odom_msg->pose.pose.position.y = msg->position[1];
    odom_msg->pose.pose.position.z = msg->position[2];
    odom_msg->pose.pose.orientation.x = msg->imu_state.quaternion[0];
    odom_msg->pose.pose.orientation.y = msg->imu_state.quaternion[1];
    odom_msg->pose.pose.orientation.z = msg->imu_state.quaternion[2];
    odom_msg->pose.pose.orientation.w = msg->imu_state.quaternion[3];

    // Fülle Geschwindigkeiten
    odom_msg->twist.twist.linear.x = msg->velocity[0];
    odom_msg->twist.twist.linear.y = msg->velocity[1];
    odom_msg->twist.twist.linear.z = msg->velocity[2];
    odom_msg->twist.twist.angular.x = msg->imu_state.gyroscope[0];
    odom_msg->twist.twist.angular.y = msg->imu_state.gyroscope[1];
    odom_msg->twist.twist.angular.z = msg->imu_state.gyroscope[2];

    odom_pub_->publish(std::move(odom_msg));
}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimOdomPublisher>());
    rclcpp::shutdown();
    return 0;
}