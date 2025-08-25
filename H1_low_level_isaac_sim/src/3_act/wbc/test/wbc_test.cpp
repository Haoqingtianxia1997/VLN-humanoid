#include "gtest/gtest.h"
#include "whole_body_controller.h" // Your WBC header

#include <fstream>
#include <stdexcept>
#include <filesystem>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <ament_index_cpp/get_package_prefix.hpp>

// Test fixture for the WholeBodyController tests
class WBCTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = WBCConfig();
        config.urdf_path = "/home/argo/programmier_stuff/UNI/ip/src/models/unitree_ros/robots/h1_description/urdf/h1.urdf";
        std::cout << config.urdf_path << std::endl;
        config.contact_frame_names = {"left_ankle_link", "right_ankle_link"};
        
        wbc = std::make_unique<WholeBodyController>(config);
    }

    void TearDown() override {
        wbc.reset();
    }

    std::unique_ptr<WholeBodyController> wbc;
    WBCConfig config;
};

// Test case for the constructor
TEST_F(WBCTest, Constructor) {
    // Check if the WBC object was created
    ASSERT_NE(wbc, nullptr);

    // nq should be 20 (13 joints + 6 floating base quaternion)
    // nv should be 19 (13 joints + 6 floating base quaternion)
    EXPECT_EQ(wbc->get_nq(), 19);
    EXPECT_EQ(wbc->get_nv(), 19);
}

// Test case for the solve method
TEST_F(WBCTest, Solve) {
    // Create some dummy data for the solve method
    Eigen::VectorXd q_current = Eigen::VectorXd::Zero(wbc->get_nq());
    Eigen::VectorXd v_current = Eigen::VectorXd::Zero(wbc->get_nv());

    WBCReferenceData references;
    references.a_com_des = Eigen::Vector3d::Zero();
    references.a_ang_des = Eigen::Vector3d::Zero();
    references.f_c_des = Eigen::VectorXd::Zero(3 * config.contact_frame_names.size());


    // Call the solve method
    bool success = wbc->solve(q_current, v_current, references);

    // Check if the solve method returns true
    EXPECT_TRUE(success);

    // Check the output torques (for this basic test, we assume they are not NaN)
    const auto& torques = wbc->get_joint_torques();
    ASSERT_FALSE(torques.hasNaN());
}
