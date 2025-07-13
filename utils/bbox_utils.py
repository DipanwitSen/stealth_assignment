from typing import Tuple, List

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

def get_bbox_width(bbox: List[float]) -> int:
    x1, _, x2, _ = bbox
    return int(x2 - x1)

def get_foot_position(bbox: List[float]) -> Tuple[int, int]:
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx ** 2 + dy ** 2) ** 0.5

def measure_xy_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    return p1[0] - p2[0], p1[1] - p2[1]
