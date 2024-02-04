class ModelConfig:
    POSE_CONFIG = "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    POSE_CHECKPOINT = "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"
    DET_CONFIG = "rtmdet_m_8xb32-300e_coco.py"
    DET_CHECKPOINT = "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"

class IndexConfig:
    BODY_IDX_START = 0
    BODY_IDX_END = 16
    FOOT_IDX_START = 17
    FOOT_IDX_END = 22
    FACE_IDX_START = 23
    FACE_IDX_END = 90
    HAND_IDX_START = 91
    HAND_IDX_END = 132

    WHOLEBODY_CONNECTIONS = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
        [15, 17],
        [15, 18],
        [15, 19],
        [16, 20],
        [16, 21],
        [16, 22],
    ]
