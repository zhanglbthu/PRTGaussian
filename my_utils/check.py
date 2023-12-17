from PIL import Image

def is_image_black(image_path):
    """
    Check if the given image is completely black.
    :param image_path: Path to the image file.
    :return: True if the image is completely black, False otherwise.
    """
    with Image.open(image_path) as img:
        # Convert the image to grayscale for simplicity
        img = img.convert("L")
        width, height = img.size

        # Check each pixel
        for x in range(width):
            for y in range(height):
                if img.getpixel((x, y)) != 0:
                    # Found a non-black pixel
                    return False

        # All pixels are black
        return True

# Example usage
image_path = "env_map.jpg"  # Replace with your image path
if is_image_black(image_path):
    print("The image is completely black.")
else:
    print("The image is not completely black.")
    # 打印不为0的像素点
    with Image.open(image_path) as img:
        # Convert the image to grayscale for simplicity
        img = img.convert("L")
        width, height = img.size

        # Check each pixel
        for x in range(width):
            for y in range(height):
                if img.getpixel((x, y)) != 0:
                    # Found a non-black pixel
                    print(x, y, img.getpixel((x, y)))
