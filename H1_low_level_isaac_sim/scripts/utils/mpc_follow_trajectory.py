import numpy as np  
import casadi as ca  
import matplotlib.pyplot as plt  # Import plotting library
import matplotlib  # Set matplotlib configuration
from matplotlib import animation  # Animation functionality
from matplotlib.patches import FancyArrowPatch  # For drawing arrows
import time  # For timing
import omni

# ============ MPC Trajectory Tracking Parameters and Functions ============
# MPC parameters
N = 5   # Prediction horizon
dt = 0.05     # Control time step

# Cost function weights
Q_x = 1000     # Position x error weight
Q_y = 1000     # Position y error weight  
Q_theta = 1000000  # Orientation error weight
R_vx = 0.1     # x-direction velocity control penalty
R_vy = 100    # y-direction velocity control penalty (suppress lateral slip)
R_w = 0.00000001    # Angular velocity control penalty

def generate_track(scale=3.0):
    """Generate test trajectory with the 150th point at origin (0,0)"""
    num_points = 300
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # Generate basic trajectory
    x = scale * np.cos(t) + 1.5 * np.sin(2 * t)
    y = scale * np.sin(t) + 1.0 * np.cos(3 * t)
    
    # Ensure the 150th point (index 149) is at origin
    target_idx = 149  # 150th point (0-indexed)
    
    # Calculate offset needed to move 150th point to origin
    x_offset = x[target_idx]
    y_offset = y[target_idx]
    
    # Apply offset to entire trajectory
    x = x - x_offset
    y = y - y_offset
    
    print(f"Trajectory generated: 150th point at ({x[target_idx]:.6f}, {y[target_idx]:.6f})")
    print(f"Trajectory range: x=[{x.min():.2f}, {x.max():.2f}], y=[{y.min():.2f}, {y.max():.2f}]")
    
    return np.column_stack((x, y))

def mpc_control(state, trajectory):
    """MPC controller"""
    try:
        opti = ca.Opti()  # Create optimizer
        distances = np.linalg.norm(trajectory[:, :2] - state[:2], axis=1)  # Distance from current point to trajectory
        nearest_idx = np.argmin(distances)  # Find nearest point index
        ref_points = trajectory[nearest_idx:nearest_idx + N + 1]  # Get prediction points

        if len(ref_points) < N + 1:  # If not enough, pad with last point
            pad = np.tile(ref_points[-1], (N + 1 - len(ref_points), 1))
            ref_points = np.vstack((ref_points, pad))

        dx_array = ref_points[1:, 0] - ref_points[:-1, 0]  # Calculate direction by differentiation
        dy_array = ref_points[1:, 1] - ref_points[:-1, 1]
        theta_refs = np.unwrap(np.arctan2(dy_array, dx_array))  # Calculate and unwrap reference orientation

        X = opti.variable(3, N + 1)  # State variables
        U = opti.variable(3, N)      # Control variables
        opti.subject_to(X[:, 0] == state)  # Initial state constraint

        cost = 0  # Initialize cost
        for k in range(N):
            x_ref, y_ref = ref_points[k, :2]  # Current reference point position
            theta_ref = theta_refs[k]         # Current reference point orientation
            theta_err = ca.atan2(ca.sin(X[2, k] - theta_ref), ca.cos(X[2, k] - theta_ref))  # Handle orientation wrap-around

            # Build cost function
            cost += Q_x * (X[0, k] - x_ref) ** 2
            cost += Q_y * (X[1, k] - y_ref) ** 2
            cost += Q_theta * theta_err ** 2
            cost += R_vx * U[0, k] ** 2 + R_vy * U[1, k] ** 2 + R_w * U[2, k] ** 2

            # System dynamics constraints
            theta_k = X[2, k]
            dx = (U[0, k] * ca.cos(theta_k) - U[1, k] * ca.sin(theta_k)) * dt
            dy = (U[0, k] * ca.sin(theta_k) + U[1, k] * ca.cos(theta_k)) * dt
            dtheta = U[2, k] * dt
            opti.subject_to(X[:, k + 1] == X[:, k] + ca.vertcat(dx, dy, dtheta))

        opti.minimize(cost)  # Set optimization objective

        # Control input constraints
        opti.subject_to(opti.bounded(-1.0, U[0, :], 1.0))
        opti.subject_to(opti.bounded(-0.2, U[1, :], 0.2))
        opti.subject_to(opti.bounded(-0.7, U[2, :], 0.7))
        opti.subject_to(U[0, :] >= 0)  # Prohibit backward motion

        opts = {'ipopt.print_level': 0, 'print_time': 0}  # Solver options
        opti.solver('ipopt', opts)

        sol = opti.solve()  # Try to solve
        return sol.value(U[:, 0])  # Return first step control
    except:
        print("MPC solving failed, using default control")
        return np.array([0.5, 0.0, 0.0])  # Default forward motion

def get_robot_state(robot):
    """Get robot state from Isaac environment [x, y, theta]"""
    # Get robot's root body position and orientation
    root_pos = robot.data.root_pos_w[0].cpu().numpy()  # [x, y, z]
    root_quat = robot.data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
    root_pose = np.array([root_pos[0], root_pos[1], root_pos[2], root_quat[0], root_quat[1], root_quat[2], root_quat[3]])
    # Calculate yaw angle from quaternion
    w, x, y, z = root_quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return np.array([root_pos[0], root_pos[1], yaw]), root_pose

def create_trajectory_visualization(env, trajectory, stride=5):
    """Create trajectory visualization spheres in Isaac environment"""
    try:
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdGeom, Gf, UsdShade
        
        print(f"🎨 Creating trajectory visualization with {len(trajectory)} points (stride={stride})")
        
        # Create parent prim for trajectory visualization
        trajectory_prim_path = "/World/TrajectoryVisualization"
        prim_utils.create_prim(trajectory_prim_path, "Xform")
        
        # Create spheres for trajectory points
        for i, point in enumerate(trajectory[::stride]):
            sphere_path = f"{trajectory_prim_path}/sphere_{i}"
            
            # Create sphere geometry
            sphere_prim = prim_utils.create_prim(
                sphere_path,
                "Sphere",
                position=[point[0], point[1], 0.05],  # z=0.05 to float slightly above ground
                scale=[0.1, 0.1, 0.1]  # Small sphere
            )
            
            # Set sphere material to red
            if sphere_prim:
                # Create material
                material_path = f"{sphere_path}/material"
                material = UsdShade.Material.Define(env.scene.stage, material_path)
                
                # Create shader
                shader = UsdShade.Shader.Define(env.scene.stage, f"{material_path}/shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", "color3f").Set((1.0, 0.2, 0.2))  # Red color
                shader.CreateInput("roughness", "float").Set(0.4)
                shader.CreateInput("metallic", "float").Set(0.0)
                
                # Connect material and shader
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                
                # Bind material to sphere
                UsdShade.MaterialBindingAPI(sphere_prim).Bind(material)
        
        print(f"✅ Trajectory visualization created with {len(trajectory[::stride])} spheres")
        
    except Exception as e:
        print(f"⚠️ Advanced visualization failed: {e}")
        print("🔄 Trying simple visualization method...")
        create_simple_trajectory_visualization(env, trajectory, stride)

def create_simple_trajectory_visualization(env, trajectory, stride=5):
    """Create simple trajectory visualization"""
    try:
        from pxr import UsdGeom, Gf
        
        print(f"🎨 Creating simple trajectory visualization with {len(trajectory)} points (stride={stride})")
        
        stage = env.scene.stage
        
        # Create parent prim for trajectory visualization
        trajectory_prim = stage.DefinePrim("/World/TrajectoryVisualization", "Xform")
        
        # Create simple spheres for trajectory points
        for i, point in enumerate(trajectory[::stride]):
            sphere_path = f"/World/TrajectoryVisualization/sphere_{i}"
            
            # Create sphere
            sphere_prim = stage.DefinePrim(sphere_path, "Sphere")
            
            # Set position
            xform_api = UsdGeom.XformCommonAPI(sphere_prim)
            xform_api.SetTranslate((point[0], point[1], 0.05))
            xform_api.SetScale((0.1, 0.1, 0.1))
            
            # Set color attribute
            sphere_geom = UsdGeom.Sphere(sphere_prim)
            color_attr = sphere_geom.CreateDisplayColorAttr()
            color_attr.Set([(1.0, 0.2, 0.2)])  # Red color
        
        print(f"✅ Simple trajectory visualization created with {len(trajectory[::stride])} spheres")
        
    except Exception as e:
        print(f"❌ Simple visualization also failed: {e}")
        print("   Continuing without trajectory visualization...")

def print_trajectory_info(trajectory):
    """Print trajectory information to console"""
    print("📊 Trajectory Information:")
    print(f"   Points: {len(trajectory)}")
    print(f"   X range: [{np.min(trajectory[:, 0]):.2f}, {np.max(trajectory[:, 0]):.2f}]")
    print(f"   Y range: [{np.min(trajectory[:, 1]):.2f}, {np.max(trajectory[:, 1]):.2f}]")
    
    # Calculate total trajectory length
    distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    total_length = np.sum(distances)
    print(f"   Total length: {total_length:.2f} m")
    
    # Print key points
    print("   Key points:")
    for i in [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, -1]:
        x, y = trajectory[i]
        print(f"     Point {i}: ({x:.2f}, {y:.2f})")

def update_robot_trajectory_visualization(env, robot_path, max_points=100):
    """Update robot trajectory visualization, showing the path the robot has traveled"""
    # This function can be used to display the actual trajectory traveled by the robot
    # Can be implemented in future versions
    pass
