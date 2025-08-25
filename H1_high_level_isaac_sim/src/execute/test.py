from scipy.spatial.transform import Rotation as R
import numpy as np

scalar = 5
quat = np.array([0.14405515789985657, 0.08654874563217163, -0.5931553244590759, 0.7873526215553284])

rot = R.from_quat(quat, scalar_first=True)

print(scalar * np.diagonal(rot.as_matrix()))
euler_deg = rot.as_euler('xyz', degrees=True)
print("Euler angles (deg, sequence='xyz'):", euler_deg)