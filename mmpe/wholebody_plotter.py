import cv2
import numpy as np
from mmpe.wholebody_config import (
    BODY_IDX_END,
    BODY_IDX_START,
    FACE_IDX_END,
    FACE_IDX_START,
    FOOT_IDX_END,
    FOOT_IDX_START,
    HAND_IDX_END,
    HAND_IDX_START,
    WHOLEBODY_CONNECTIONS,
)


def plot_keypoints_and_connections(img: np.ndarray, keypoints: np.ndarray):
    img_copy = img.copy()

    img_copy = plot_keypoints(img_copy, keypoints)
    img_copy = plot_connections(img_copy, keypoints)

    return img_copy


def plot_keypoints(img: np.ndarray, keypoints: np.ndarray):
    img_copy = img.copy()

    # Plot keypoints
    for idx in range(BODY_IDX_START, BODY_IDX_END + 1):
        x, y = keypoints[idx]
        cv2.circle(img_copy, (int(x), int(y)), 5, (255, 0, 0), -1)
    for idx in range(FACE_IDX_START, FACE_IDX_END + 1):
        x, y = keypoints[idx]
        cv2.circle(img_copy, (int(x), int(y)), 2, (0, 255, 0), -1)
    for idx in range(HAND_IDX_START, HAND_IDX_END + 1):
        x, y = keypoints[idx]
        cv2.circle(img_copy, (int(x), int(y)), 2, (0, 0, 255), -1)
    for idx in range(FOOT_IDX_START, FOOT_IDX_END + 1):
        x, y = keypoints[idx]
        cv2.circle(img_copy, (int(x), int(y)), 5, (255, 255, 0), -1)

    return img_copy


def plot_connections(img: np.ndarray, keypoints: np.ndarray):
    img_copy = img.copy()

    # Plot connections
    for connection in WHOLEBODY_CONNECTIONS:
        x1, y1 = keypoints[connection[0]]
        x2, y2 = keypoints[connection[1]]
        cv2.line(
            img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2
        )

    return img_copy
