"""
risk_engine.py

Collision Risk Engine Module
-----------------------------
Calculates collision risk status based on Distance, Position, Movement, Relative Speed, and TTC.
"""

import numpy as np
from distance_estimation import estimate_distance

try:
    import cv2
except ImportError:
    cv2 = None

PATH_ZONE_COLOR = (255, 140, 20)      # Bright Blue
NEUTRAL_ZONE_COLOR = (120, 120, 120)  # Gray
OBJECT_BOX_COLOR = (0, 255, 255)     # Yellow
CRITICAL_BOX_COLOR = (0, 0, 255)     # Red


def check_in_path(center_x, bottom_y, frame_width, frame_height):
    """Determine whether an object is inside the V-shaped driving path."""
    bottom_screen_y = frame_height
    top_y = frame_height * 0.5  # Horizon

    bottom_left_x = frame_width * 0.40
    bottom_right_x = frame_width * 0.60

    top_left_x = frame_width * 0.30
    top_right_x = frame_width * 0.70

    if bottom_y < top_y:
        bottom_y = top_y

    ratio = (bottom_y - top_y) / (bottom_screen_y - top_y)
    current_left_x = top_left_x + ratio * (bottom_left_x - top_left_x)
    current_right_x = top_right_x + ratio * (bottom_right_x - top_right_x)

    return current_left_x <= center_x <= current_right_x


def is_approaching(previous_distance, current_distance):
    return current_distance < previous_distance


def calculate_relative_speed(previous_distance, current_distance, delta_time):
    if delta_time <= 0:
        return 0.0
    return (previous_distance - current_distance) / delta_time


def calculate_ttc(distance, relative_speed):
    if relative_speed <= 0:
        return None
    return distance / relative_speed


def distance_risk(distance):
    if distance < 8:
        return 21    # كانت 30
    elif distance < 15:
        return 14    # كانت 20
    elif distance < 30:
        return 7     # كانت 10
    elif distance < 50:
        return 3.5   # كانت 5
    return 0


def position_risk(in_path):
    return 17.5 if in_path else 0  # كانت 25


def movement_risk(approaching):
    return 10.5 if approaching else 0  # كانت 15


def speed_risk(relative_speed):
    if relative_speed > 10:
        return 14    # كانت 20
    elif relative_speed > 5:
        return 7     # كانت 10
    elif relative_speed > 2:
        return 3.5   # كانت 5
    return 0


def ttc_risk(ttc):
    if ttc is None:
        return 0
    if ttc < 1.5:
        return 24.5  # كانت 35
    elif ttc < 3:
        return 17.5  # كانت 25
    elif ttc < 5:
        return 7     # كانت 10
    return 0


def calculate_risk_score(distance, in_path, approaching, relative_speed):
    score = 0
    score += distance_risk(distance)
    score += position_risk(in_path)
    score += movement_risk(approaching)
    score += speed_risk(relative_speed)

    ttc = calculate_ttc(distance, relative_speed)
    score += ttc_risk(ttc)

    return min(score, 100), ttc


def get_risk_status(score):
    if score <= 25:
        return "SAFE"
    elif score <= 50:
        return "WARNING"
    elif score <= 75:
        return "HIGH RISK"
    return "CRITICAL / BRAKE"


def process_object(object_data, frame_width=1280, frame_height=720):
    distance = object_data["distance"]
    previous_distance = object_data.get("previous_distance", distance)
    center_x = object_data["center_x"]
    bottom_y = object_data.get("bottom_y", frame_height)
    timestamp = object_data.get("timestamp", 0.0)
    previous_timestamp = object_data.get("previous_timestamp", timestamp)

    delta_time = timestamp - previous_timestamp
    if delta_time <= 0:
        delta_time = 1e-6

    in_path = check_in_path(center_x, bottom_y, frame_width, frame_height)
    approaching = is_approaching(previous_distance, distance)
    relative_speed = calculate_relative_speed(previous_distance, distance, delta_time)

    score, ttc = calculate_risk_score(distance, in_path, approaching, relative_speed)
    status = get_risk_status(score)

    return {
        "track_id": object_data.get("track_id"),
        "class": object_data.get("class"),
        "distance": round(distance, 2),
        "in_path": in_path,
        "approaching": approaching,
        "relative_speed": round(relative_speed, 2),
        "ttc": round(ttc, 2) if ttc is not None else None,
        "risk_score": score,
        "status": status,
    }


def process_object_from_bbox(detection, focal_length, frame_width=1280, frame_height=720, previous_state=None):
    x1, y1, x2, y2 = detection["bbox"]
    center_x = (x1 + x2) / 2
    bottom_y = y2

    distance = estimate_distance(
        object_class=detection.get("class"),
        bbox=detection["bbox"],
        focal_length=focal_length,
    )

    if distance is None:
        return None

    previous_distance = previous_state["distance"] if previous_state else distance
    previous_timestamp = previous_state["timestamp"] if previous_state else detection.get("timestamp", 0.0)

    object_data = {
        "track_id": detection.get("track_id"),
        "class": detection.get("class"),
        "distance": distance,
        "center_x": center_x,
        "bottom_y": bottom_y,
        "previous_distance": previous_distance,
        "timestamp": detection.get("timestamp", 0.0),
        "previous_timestamp": previous_timestamp,
    }

    result = process_object(object_data, frame_width=frame_width, frame_height=frame_height)
    result["timestamp"] = detection.get("timestamp", 0.0)

    return result


def draw_path_zone(frame, frame_width, alpha=0.15):
    if cv2 is None:
        raise ImportError("opencv-python is required for draw_path_zone()")

    height = frame.shape[0]

    bottom_y = height
    bottom_left_x = int(frame_width * 0.40)
    bottom_right_x = int(frame_width * 0.60)

    top_y = int(height * 0.5)
    top_left_x = int(frame_width * 0.30)
    top_right_x = int(frame_width * 0.70)

    pts = np.array([
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], PATH_ZONE_COLOR)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.line(frame, (bottom_left_x, bottom_y), (top_left_x, top_y), PATH_ZONE_COLOR, 2)
    cv2.line(frame, (bottom_right_x, bottom_y), (top_right_x, top_y), PATH_ZONE_COLOR, 2)
    cv2.putText(frame, "PATH", (bottom_left_x + 10, bottom_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PATH_ZONE_COLOR, 2)

    return frame


def draw_object_result(frame, bbox, result):
    if cv2 is None:
        raise ImportError("opencv-python is required for draw_object_result()")

    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = CRITICAL_BOX_COLOR if result["status"] == "CRITICAL / BRAKE" else OBJECT_BOX_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f'{result["class"]} {result["distance"]}m {result["status"]}'
    cv2.putText(frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame