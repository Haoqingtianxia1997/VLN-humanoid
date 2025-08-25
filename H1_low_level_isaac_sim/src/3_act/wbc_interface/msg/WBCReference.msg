std_msgs/Header header

# Desired linear acceleration of the CoM
geometry_msgs/Vector3 a_com_des

# Desired angular acceleration of the CoM
geometry_msgs/Vector3 a_ang_des

# Desired contact forces (flattened vector)
# Example for 2 contacts: [fx1, fy1, fz1, fx2, fy2, fz2]
float64[] f_c_des