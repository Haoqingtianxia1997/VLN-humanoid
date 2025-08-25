#ifndef SIM_ODOM_PUBLISHER_NODE_HPP_
#define SIM_ODOM_PUBLISHER_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"
#include "nav_msgs/msg/odometry.hpp"


class SimOdomPublisher : public rclcpp::Node {
public:
    SimOdomPublisher();

private:
    void sportModeCallback(const unitree_go::msg::SportModeState::SharedPtr msg) const;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr sport_mode_state_sub_;
};

#endif // SIM_ODOM_PUBLISHER_NODE_HPP_