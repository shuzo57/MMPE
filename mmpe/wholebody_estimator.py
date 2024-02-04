import os

import numpy as np
from mmpe.mmpose_estimator import MMPoseEstimator
from mmpe.wholebody_config import IndexConfig, ModelConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

POSE_CONFIG = os.path.join(parent_dir, "models", ModelConfig.POSE_CONFIG)
POSE_CHECKPOINT = os.path.join(parent_dir, "models", ModelConfig.POSE_CHECKPOINT)
DET_CONFIG = os.path.join(parent_dir, "models", ModelConfig.DET_CONFIG)
DET_CHECKPOINT = os.path.join(parent_dir, "models", ModelConfig.DET_CHECKPOINT)

class WholeBodyEstimator(MMPoseEstimator):
    def __init__(
        self,
        pose_config: str = POSE_CONFIG,
        pose_checkpoint: str = POSE_CHECKPOINT,
        det_config: str = DET_CONFIG,
        det_checkpoint: str = DET_CHECKPOINT,
    ):
        super().__init__(
            pose_config, pose_checkpoint, det_config, det_checkpoint
        )

    def get_body_keypoints(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトの体のキーポイントを返します。

        Args:
            obj_idx (int, optional): キーポイントを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトの体のキーポイント。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.keypoints[0][
            IndexConfig.BODY_IDX_START : IndexConfig.BODY_IDX_END + 1  # noqa: E203
        ]

    def get_foot_keypoints(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトの足のキーポイントを返します。

        Args:
            obj_idx (int, optional): キーポイントを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトの足のキーポイント。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.keypoints[0][
            IndexConfig.FOOT_IDX_START : IndexConfig.FOOT_IDX_END + 1  # noqa: E203
        ]

    def get_face_keypoints(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトの顔のキーポイントを返します。

        Args:
            obj_idx (int, optional): キーポイントを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトの顔のキーポイント。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.keypoints[0][
            IndexConfig.FACE_IDX_START : IndexConfig.FACE_IDX_END + 1  # noqa: E203
        ]

    def get_hand_keypoints(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトの手のキーポイントを返します。

        Args:
            obj_idx (int, optional): キーポイントを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトの手のキーポイント。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.keypoints[0][
            IndexConfig.HAND_IDX_START : IndexConfig.HAND_IDX_END + 1  # noqa: E203
        ]
