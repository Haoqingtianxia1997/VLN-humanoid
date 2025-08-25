#ifndef REFERENCE_GENERATOR_NODE_H
#define REFERENCE_GENERATOR_NODE_H

#include <array>
#include <vector>
#include <tuple>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "mpc_interface/msg/y_reference.hpp" 
#include "reference_generator.hpp"


class ReferenceGeneratorNode : public rclcpp::Node {
private:
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr cmd_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_twist_sub_;
    rclcpp::Publisher<mpc_interface::msg::YReference>::SharedPtr yref_pub_;

public:
    ReferenceGeneratorNode();

private:
    void pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void twist_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void publish_yref(
        std::vector<std::array<double, SRBM_ROBOT_NY>> yrefs, 
        std::array<double, SRBM_ROBOT_NYN> yref_e
    );

};

#endif // REFERENCE_GENERATOR_NODE_H