import numpy as np
from mmpe.mmpose_estimator import MMPoseEstimator
from mmpe.wholebody_config import (
    BODY_IDX_END,
    BODY_IDX_START,
    FACE_IDX_END,
    FACE_IDX_START,
    FOOT_IDX_END,
    FOOT_IDX_START,
    HAND_IDX_END,
    HAND_IDX_START,
)


class WholeBodyEstimator(MMPoseEstimator):
    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
        det_config: str,
        det_checkpoint: str,
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
            BODY_IDX_START : BODY_IDX_END + 1  # noqa: E203
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
            FOOT_IDX_START : FOOT_IDX_END + 1  # noqa: E203
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
            FACE_IDX_START : FACE_IDX_END + 1  # noqa: E203
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
            HAND_IDX_START : HAND_IDX_END + 1  # noqa: E203
        ]
