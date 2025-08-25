import pinocchio as pin
import os

H1_CONTACT_LINK_NAMES = ["left_ankle_link", "right_ankle_link"] 

def get_robot_inertial_properties(model_path: str, contact_link_names_to_extract: list[str]) -> tuple:
    """
    Loads a robot model from URDF and computes its composite
    rigid body mass and inertia tensor about its center of mass.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The URDF file was not found at: {model_path}")

    try:
        model = pin.buildModelFromUrdf(model_path)
    except Exception as e:
        print(f"Failed to load the model from {model_path}.")
        print(f"Error: {e}")
        return None, None, None, None, None, None
    
    data = model.createData()
    # Use a neutral configuration for calculations
    q = pin.neutral(model)
    v = pin.utils.zero(model.nv) # Zero velocities

    # Compute the forward kinematics to get the CoM and inertia
    pin.ccrba(model, data, q, v)

    # Compute joint space inertia matrix (CRBA)
    # M_joint_space = pin.crba(model, data, q)

    # Compute the total mass and inertia
    pin.computeCentroidalMomentum(model, data, q, v) 

    total_mass = data.mass[0]
    com_position = data.com[0].copy() # Position of the CoM
    
    Ig_total_robot = data.Ig.inertia.copy()

    contact_properties = {}
    if contact_link_names_to_extract:
        for link_name in contact_link_names_to_extract:
            if model.existFrame(link_name):
                frame_id = model.getFrameId(link_name)
                # The parent_joint_id is the ID of the joint that moves the link/body
                # to which this frame is attached. model.inertias is indexed by joint_id.
                parent_joint_id = model.frames[frame_id].parentJoint 
                
                # model.inertias[0] is for the universe.
                # Actual link inertias correspond to joint indices 1 to model.njoints-1.
                if 0 < parent_joint_id < model.njoints:
                    link_pin_inertia_obj = model.inertias[parent_joint_id]
                    contact_properties[link_name] = {
                        'mass': link_pin_inertia_obj.mass,
                        # .lever is the CoM of the link, expressed in the link's own frame
                        'com_in_link_frame': link_pin_inertia_obj.lever.copy(), 
                        # .inertia is the 3x3 inertia tensor of the link about its CoM, 
                        # expressed in the link's own frame
                        'inertia_at_link_com': link_pin_inertia_obj.inertia.copy() 
                    }
                else:
                    print(f"Warning: Parent joint ID {parent_joint_id} for frame '{link_name}' is out of expected range for a link. Skipping.")
            else:
                print(f"Warning: Contact link/frame name '{link_name}' not found in model. Skipping.")

    return model, data, total_mass, com_position, Ig_total_robot, contact_properties

def get_h1_robot_inertial_properties(urdf_path: str) -> tuple:
    """
    Loads the H1 robot model and computes its inertial properties.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Die URDF-Datei wurde unter dem angegebenen Pfad nicht gefunden: {urdf_path}")

    _, _, total_mass, com, inertia_tensor, contact_properties = get_robot_inertial_properties(urdf_path, H1_CONTACT_LINK_NAMES)
    return total_mass, com, inertia_tensor, contact_properties


if __name__ == "__main__":
    import os
    base_dir = os.path.join(os.getenv('HOME'), 'programmier_stuff', 'UNI', 'ip', 'src', 'models')
    total_mass, com, inertia_tensor, contact_properties = get_h1_robot_inertial_properties(base_dir)
    print(f"Total Robot Mass: {total_mass:.4f} kg")
    print(f"Center of Mass (CoM): {com}")
    print("Inertia Tensor about CoM (I_g):")
    print(inertia_tensor)
    print("Contact Properties:")
    for link_name, props in contact_properties.items():
        print(f"  {link_name}:")
        for prop_name, prop_value in props.items():
            print(f"    {prop_name}: {prop_value}")