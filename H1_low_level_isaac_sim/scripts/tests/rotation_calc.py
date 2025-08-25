import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# camera facing upwards
q_orig = [0, 0, -0.7071, 0.7071]          # (x, y, z, w)

angle  = math.radians(-100)                # -100° → rad
axis   = np.array([1.0, 0.0, 0.0])       # local x-axis
r_delta = R.from_rotvec(angle * axis)

q_new = (R.from_quat(q_orig) * r_delta).as_quat()
print(q_new)
