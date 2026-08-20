"""
distance_estimation.py

Monocular Distance Estimation Module
------------------------------------
Estimates real-world distance to a detected object using a single camera,
based on the object's known real-world width and its bounding-box width
in pixels (similar-triangles method).
"""
import math

KNOWN_WIDTHS = {
    "Person": 0.5,
    "Bicycle": 0.6,
    "Car": 1.8,
    "Motorcycle": 0.8,
    "Bus": 2.5,
    "Truck": 2.5,
}


def calculate_focal_length(known_distance, real_width, pixel_width):
    if real_width <= 0:
        raise ValueError("real_width must be greater than 0")

    return (pixel_width * known_distance) / real_width


def approximate_focal_length(image_width_px, fov_degrees=60):
    fov_rad = math.radians(fov_degrees)
    return image_width_px / (2 * math.tan(fov_rad / 2))


def estimate_distance(object_class, bbox, focal_length, known_widths=None):
    widths = known_widths or KNOWN_WIDTHS

    real_width = widths.get(object_class)
    if real_width is None:
        return None

    x1, _, x2, _ = bbox
    pixel_width = x2 - x1

    if pixel_width <= 0:
        return None

    return (real_width * focal_length) / pixel_width