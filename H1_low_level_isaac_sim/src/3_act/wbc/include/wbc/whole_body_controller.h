#ifndef WHOLE_BODY_CONTROLLER
#define WHOLE_BODY_CONTROLLER

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <optional>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <proxsuite/proxqp/dense/dense.hpp>

namespace wbc
{

// A simple struct to pass reference data to the solver
struct WBCReferenceData
{
    Eigen::Vector3d a_com_des;
    Eigen::Vector3d a_ang_des; // Desired angular acceleration of the base
    Eigen::VectorXd f_c_des;   // Stacked vector of desired contact forces
};

struct WBCConfig {
    std::string urdf_path;
    std::vector<std::string> contact_frame_names;
    double w_com = 1.0;
    double w_force = 1e-3;
    double w_reg = 1e-4;
    double w_contact= 1e6;
    double friction_mu = 0.7;
};

class WholeBodyController {
private:
    WBCConfig config_; 

    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<pinocchio::FrameIndex> contact_frame_ids_;
    int nq_, nv_, num_contacts_, ineq_constraints_, eq_constraints_, dim_;
    double mass_;

    // QP Solver
    std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp_;

    // Output torques
    Eigen::VectorXd tau_out_;

public:
    WholeBodyController(const WBCConfig& config);

    void update_config(const WBCConfig& config);

    // Main solver function
    bool solve(const Eigen::VectorXd& q_current, const Eigen::VectorXd& v_current, const WBCReferenceData& references);

    // Getter for the result
    pinocchio::SE3 get_frame_placement(const std::string& frame_name);
    const Eigen::VectorXd& get_joint_torques() const { return this->tau_out_; }

    // Getter for the parameters
    int get_nq() const { return this->nq_; }
    int get_nv() const { return this->nv_; }
    int get_num_contacts() const { return this->num_contacts_; }
    double get_total_mass() const;
    double get_z_gravity() const { return std::abs(this->model_.gravity.linear().z()); }
    Eigen::Vector3d get_com_position() const { return this->data_.com[0]; }
    Eigen::Vector3d get_com_velocity() const { return this->data_.vcom[0]; }

private:
    void setup_qp();
};

} // namespace wbc_controller

#endif  // WHOLE_BODY_CONTROLLER