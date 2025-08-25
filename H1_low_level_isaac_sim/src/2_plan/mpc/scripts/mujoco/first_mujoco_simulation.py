if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    import time
    import os
    import numpy as np
    

    # Replace with the actual path to your XML file
    MODELS_DIR = os.path.join(os.getcwd(), 'src', '0_common')
    model_path = os.path.join(MODELS_DIR, 'robot_launch/config/h1/h1_description/mjcf/scene.xml')
    
    try:
        m = mujoco.MjModel.from_xml_path(model_path)
        d = mujoco.MjData(m)

        KP = 200.0
        KD = 20.0
        KI = 1.0
        MAX_INTEGRAL = 10.0
        integral_error = np.zeros(m.nu)

        try:
            key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
            if key_id == -1:
                raise ValueError("Keyframe 'home' wurde im Modell nicht gefunden.")
            
            qpos_home = m.key_qpos[key_id]
            qvel_home = m.key_qvel[key_id]
            d.qpos[:] = qpos_home
            d.qvel[:] = qvel_home
            print("INFO: Simulationszustand erfolgreich auf den Keyframe 'home' gesetzt.")
        except ValueError as e:
            print(f"WARNUNG: Keyframe konnte nicht gesetzt werden: {e}")

        with mujoco.viewer.launch_passive(m, d) as viewer:
            # Simulate for a few seconds
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < 30: # Run for 30 seconds
                step_start = time.time()
                for i in range(m.nu):
                    position_error = qpos_home[7+i] - d.qpos[7+i]
                    integral_error[i] += position_error * m.opt.timestep
                    integral_error[i] = np.clip(integral_error[i], -MAX_INTEGRAL, MAX_INTEGRAL)
                    velocity_error = qvel_home[6+i] - d.qvel[6+i]
                    d.ctrl[i] = (KP * position_error) + (KI * integral_error[i]) + (KD * velocity_error)
                mujoco.mj_step(m, d)
                viewer.sync()

                # Rudimentary framerate control
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Error: {e}")