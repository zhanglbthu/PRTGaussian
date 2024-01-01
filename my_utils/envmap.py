from PIL import Image
import math
from torchvision import transforms

def angle_to_uv(theta, phi):
    """
    Convert an angle to UV coordinates on the environmental map.
    theta: Polar angle (0 to pi)
    phi: Azimuthal angle (-pi to pi)
    """
    u = 0.5 + (phi / (2 * math.pi))
    v = theta / math.pi
    assert 0 <= u <= 1
    assert 0 <= v <= 1
    return u, v

def direction_to_uv(direction):
    """
    Convert a direction vector to UV coordinates on the environmental map.
    Assumes direction is a normalized 3D vector (x, y, z).
    """
    x, y, z = direction
    u = 0.5 + (math.atan2(x, z) / (2 * math.pi))
    v = 0.5 - (math.asin(y) / math.pi)
    return u, v

def create_env_map(theta=None, phi=None, size=(36, 18)):
    """
    Create an environment map with a single pixel lit according to the direction light.
    direction: A 3D direction vector for the light.
    size: Size of the environment map.
    """
    width, height = size
    env_map = Image.new("RGB", size, "black")
    pixels = env_map.load()

    u, v = angle_to_uv(theta, phi)
    
    # Convert UV coordinates to pixel coordinates
    x = int(u * width)
    y = int(v * height)
    
    # Clamp pixel coordinates to the image size
    x = min(width - 1, max(0, x))
    y = min(height - 1, max(0, y))

    # Set the pixel value
    pixels[x, y] = (255, 255, 255)  # White pixel
    env_map = transforms.ToTensor()(env_map).unsqueeze(0)
    return env_map

if __name__ == "__main__":
    # Create an environment map with a single pixel lit according to the direction light.
    env_map = create_env_map(theta=math.pi / 2, phi=math.pi / 2)
    # Save the environment map
    transforms.ToPILImage()(env_map.squeeze(0)).save("env_map.png")

