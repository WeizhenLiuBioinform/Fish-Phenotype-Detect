import math

def transform_keypoints_to_circles(keypoints_ranges, scale, rotation, rotation_center=(0, 0)):
    """Convert the range of keypoints to the circumscribed circle and apply scaling and rotation transformations"""

    def get_circle_properties(kp_range):
        """Calculate the properties of the circumscribed circle of the rectangle: center point and radius"""
        max_x, max_y, min_x, min_y = kp_range
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        radius = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2) / 2
        return center_x, center_y, radius

    def rotate_point(x, y, angle, cx, cy):
        """Rotate point around the given center point"""
        angle_rad = math.radians(angle)
        cx = cx / 4
        cy = cy / 4
        x_new = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
        y_new = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy
        return x_new, y_new

    transformed_circles = []
    for kp_range in keypoints_ranges:
        center_x, center_y, radius = get_circle_properties(kp_range)

        # Apply scaling to the radius
        radius *= scale

        # Apply rotation to the center of the circle
        center_x, center_y = rotate_point(center_x, center_y, rotation, *rotation_center)

        transformed_circles.append((center_x, center_y, radius))

    return transformed_circles

# Example keypoint ranges
keypoints_ranges = [
    (576, 432, 0, 0), (369, 301, 292, 274), (324, 160, 294, 125),
    # ... other keypoints
]

# Example scaling and rotation values
scale = 1.2  # Scaling factor
rotation = 30  # Rotation angle
rotation_center = (288, 216)  # Rotation center

# Perform the transformation
transformed_circles = transform_keypoints_to_circles(keypoints_ranges, scale, rotation, rotation_center)
print(transformed_circles)
