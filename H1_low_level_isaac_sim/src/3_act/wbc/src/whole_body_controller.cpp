#include "wbc/whole_body_controller.h"

namespace wbc
{

WholeBodyController::WholeBodyController(const WBCConfig& config) : config_(config) {
    // 1. Load Pinocchio Model
    try {
        pinocchio::urdf::buildModel(this->config_.urdf_path, this->model_);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Pinocchio model build failed: ") + e.what());
    }

    // Fallback when the URDF does not contain a free flyer joint
    if (this->model_.nq == this->model_.nv) {
        try {
            pinocchio::JointModelFreeFlyer root_joint;
            pinocchio::urdf::buildModel(this->config_.urdf_path, root_joint, this->model_);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Pinocchio model build failed: ") + e.what());
        }
    }
    this->data_ = pinocchio::Data(this->model_);

    this->nq_ = this->model_.nq;
    this->nv_ = this->model_.nv;
    this->num_contacts_ = this->config_.contact_frame_names.size();

    // 2. Get contact frame IDs
    for (const auto& name : this->config_.contact_frame_names) {
        if (!this->model_.existFrame(name)) {
            throw std::runtime_error("Contact frame not found in model: " + name);
        }
        this->contact_frame_ids_.push_back(this->model_.getFrameId(name));
    }
    
    // 3. Initialize QP solver
    setup_qp();

    // Initialize output torque vector
    this->tau_out_ = Eigen::VectorXd::Zero(this->nv_ - 6); // Only for actuated joints
}

void WholeBodyController::setup_qp() {
    // Decision variable x = [q_ddot (nv), f_c (3*num_contacts)]
    dim_ = this->nv_ + 3 * this->num_contacts_;
    // Equality constraints: contact acceleration = 0
    eq_constraints_ = 0; //3 * this->num_contacts_;
    // Inequality constraints: friction cones
    // 5 constraints per contact (4 for pyramid approx + 1 for unilateral force)
    ineq_constraints_ = 5 * this->num_contacts_;

    this->qp_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(dim_, eq_constraints_, ineq_constraints_);
}

void WholeBodyController::update_config(const WBCConfig& config) {
    this->config_ = config;
}

bool WholeBodyController::solve(const Eigen::VectorXd& q_current, const Eigen::VectorXd& v_current, const WBCReferenceData& refs) {
    // 1. UPDATE DYNAMICS WITH CURRENT STATE
    pinocchio::forwardKinematics(this->model_, this->data_, q_current, v_current);
    pinocchio::updateFramePlacements(this->model_, this->data_); // Updates the positions of all frames
    pinocchio::crba(this->model_, this->data_, q_current); // Computes the joint-space inertia matrix 
    pinocchio::nonLinearEffects(this->model_, this->data_, q_current, v_current); // Computes Coriolis, centrifugal, and gravity terms
    pinocchio::computeCentroidalMomentum(this->model_, this->data_, q_current, v_current); // Computes the time derivative of the centroidal momentum matrix

    // 2. FORMULATE THE QP
    // Hessian = Hessian (cost function), grad = gradient (cost function)
    // A_eq = Equality constraint matrix, b_eq = equality constraint vector
    // C_ineq = Inequality constraint matrix, lb/ub = inequality constraint bounds
    Eigen::MatrixXd Hessian = Eigen::MatrixXd::Zero(dim_, dim_);
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(dim_);

    // --- Cost Function ---
    // Get CoM Jacobian
    Eigen::MatrixXd J_com = this->data_.Ag;
    const Eigen::VectorXd a_com_drift = this->data_.dAg * v_current;
    
    // a) CoM Task (linear and angular)
    double w_com = this->config_.w_com;
    Eigen::MatrixXd J_com6 = J_com.topRows<6>();
    Hessian.topLeftCorner(nv_, nv_) += w_com * J_com6.transpose() * J_com6;
    Eigen::VectorXd a_des_6d(6);
    a_des_6d.head<3>() = refs.a_com_des;
    a_des_6d.tail<3>() = refs.a_ang_des;
    grad.head(nv_) += w_com * J_com6.transpose() * (a_com_drift.topRows<6>() - a_des_6d);

    // b) Contact Force Task
    double w_force = this->config_.w_force;
    Hessian.bottomRightCorner(3 * this->num_contacts_, 3 * this->num_contacts_).diagonal().array() += w_force;
    grad.tail(3 * this->num_contacts_) -= w_force * refs.f_c_des;
    
    // c) Regularization for joint accelerations
    double w_reg = this->config_.w_reg;
    Hessian.block(6, 6, nv_ - 6, nv_ - 6).diagonal().array() += w_reg;

    // --- Equality Constraints: Contact point acceleration must be zero ---
    // Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(eq_constraints_, dim_);
    // Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(eq_constraints_);
    Eigen::MatrixXd Jc_full = Eigen::MatrixXd::Zero(3 * this->num_contacts_, nv_);

    // --- Soft contact constraints ---
    double w_contact = config_.w_contact;
    for (size_t i = 0; i < this->num_contacts_; ++i) {
        pinocchio::getFrameJacobian(this->model_, this->data_, this->contact_frame_ids_[i], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
        Eigen::MatrixXd Jc = this->data_.J.topRows<3>();
        Jc_full.block(3 * i, 0, 3, nv_) = Jc;
        const pinocchio::Motion ac_drift_motion = pinocchio::getFrameClassicalAcceleration(this->model_, this->data_, contact_frame_ids_[i], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
        Eigen::Vector3d ac_drift = ac_drift_motion.linear();
        
        Hessian.topLeftCorner(nv_, nv_) += w_contact * Jc.transpose() * Jc;
        grad.head(nv_) += w_contact * Jc.transpose() * ac_drift;
    }

    // --- Inequality Constraints: Friction Cones ---
    Eigen::MatrixXd C_ineq = Eigen::MatrixXd::Zero(ineq_constraints_, dim_);
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(ineq_constraints_); // Upper bound
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(ineq_constraints_, -std::numeric_limits<double>::infinity()); // Lower bound
    
    Eigen::Matrix<double, 5, 3> G_friction;
    G_friction << 1, 0, -this->config_.friction_mu / sqrt(2),
                  -1, 0, -this->config_.friction_mu / sqrt(2),
                  0, 1, -this->config_.friction_mu / sqrt(2),
                  0, -1, -this->config_.friction_mu / sqrt(2),
                  0, 0, -1;

    for (size_t i = 0; i < this->num_contacts_; ++i)
    {
        C_ineq.block(5 * i, nv_ + 3 * i, 5, 3) = G_friction;
        ub.segment(5 * i, 5).setZero(); // C_ineq*x <= 0
        lb(5 * i + 4) = -std::numeric_limits<double>::infinity(); // Fz >= 0, so -Fz <= 0
    }
    
    // 3. SOLVE THE QP
    qp_->update(Hessian, grad, std::nullopt, std::nullopt, C_ineq, lb, ub);
    qp_->solve();

    auto& result = qp_->results;
    if (result.info.status != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
        // PROXQP_SOLVED,                           the problem is solved.
        // PROXQP_MAX_ITER_REACHED,                 the maximum number of iterations has been reached.
        // PROXQP_PRIMAL_INFEASIBLE,                the problem is primal infeasible.
        // PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE,   the closest (in L2 sense) feasible
        // PROXQP_DUAL_INFEASIBLE,                  the problem is dual infeasible.
        // PROXQP_NOT_RUN                           the solver has not been run yet.
        std::cerr << "WBC QP failed to solve! Status: " << static_cast<int>(result.info.status) << std::endl;
    }

    // 4. CALCULATE FINAL TORQUES
    Eigen::VectorXd q_ddot_star = result.x.head(nv_);
    Eigen::VectorXd f_c_star = result.x.tail(3 * this->num_contacts_);
    Eigen::VectorXd torques_from_contacts = Jc_full.transpose() * f_c_star;
    
    Eigen::VectorXd all_torques = pinocchio::rnea(this->model_, this->data_, q_current, v_current, q_ddot_star);
    this->tau_out_ = (all_torques - torques_from_contacts).tail(nv_ - 6);

    return true;
}

pinocchio::SE3 WholeBodyController::get_frame_placement(const std::string& frame_name) {
    if (!model_.existFrame(frame_name)) {
        throw std::runtime_error("Frame '" + frame_name + "' does NOT exist in the model.");
    }
    pinocchio::FrameIndex frame_id = model_.getFrameId(frame_name);
    return data_.oMf[frame_id]; 
}

double WholeBodyController::get_total_mass() const {
    if (data_.mass.empty()) {
        return pinocchio::computeTotalMass(this->model_);
    }
    return data_.mass[0];
}

} // namespace wbc