from scene.lights import Light
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

def light_to_JSON(id, light : Light):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = light.R.transpose()
    Rt[:3, 3] = light.T
    Rt[3, 3] = 1.0
    
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    light_entry = {
        "id": id,
        "type": "light",
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
    }
    return light_entry
    