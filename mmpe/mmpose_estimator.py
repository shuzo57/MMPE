import warnings

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown, init_model

warnings.filterwarnings("ignore")


class MMPoseEstimator:
    def __init__(
        self,
        pose_config: str,
        pose_checkpoint: str,
        det_config: str,
        det_checkpoint: str,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.pose_model = init_model(
            pose_config, pose_checkpoint, device=self.device
        )
        self.det_model = init_detector(
            det_config, det_checkpoint, device=self.device
        )
        self.mmdet_results = None
        self.mmpose_results = None

    def predict(self, img: np.ndarray) -> None:
        """
        画像に対して物体検出とポーズ推定を行います。

        Args:
            img (np.ndarray): 推定を行う画像。
        """
        self.mmdet_results = self.detect(img)
        class_boxes = self.process_mmdet_results(self.mmdet_results)
        self.mmpose_results = self.estimate(img, class_boxes)

    def get_bboxes(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトのバウンディングボックスを返します。

        Args:
            obj_idx (int, optional): バウンディングボックスを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトのバウンディングボックス。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.bboxes[0]

    def get_keypoints(self, obj_idx: int = 0) -> np.ndarray:
        """
        検出されたオブジェクトのキーポイントを返します。

        Args:
            obj_idx (int, optional): キーポイントを取得するオブジェクトのインデックス。デフォルトは0。

        Returns:
            np.ndarray: 指定されたオブジェクトのキーポイント。
        """
        if len(self.mmpose_results) == 0:
            return np.array([])
        return self.mmpose_results[obj_idx].pred_instances.keypoints[0]

    def detect(self, img: np.ndarray) -> DetDataSample:
        """
        画像に対して物体検出を行います。

        Args:
            img (np.ndarray): 検出を行う画像。

        Returns:
            DetDataSample: 検出されたオブジェクトのデータ。
        """
        scope = self.det_model.cfg.get("default_scope", "mmdet")
        if scope is not None:
            init_default_scope(scope)
        mmdet_results = inference_detector(self.det_model, img)
        return mmdet_results

    def estimate(self, img: np.ndarray, class_boxes: np.ndarray):
        """
        画像に対してポーズ推定を行います。

        Args:
            img (np.ndarray): ポーズ推定を行う画像。
            class_boxes (np.ndarray): ポーズ推定を行うオブジェクトのバウンディングボックス。

        Returns:
            結果を表すデータ。
        """
        scope = self.pose_model.cfg.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)
        mmpose_results = inference_topdown(self.pose_model, img, class_boxes)
        return mmpose_results

    def process_mmdet_results(
        self,
        mmdet_results: DetDataSample,
        class_id: int = 0,  # person
        class_threshold: float = 0.50,
    ) -> np.ndarray:
        """
        物体検出の結果を処理し、特定のクラスIDに属するオブジェクトのバウンディングボックスを返します。

        Args:
            mmdet_results (DetDataSample): 物体検出の結果。
            class_id (int, optional): 対象とするクラスID。デフォルトは0（人）。
            class_threshold (float, optional): クラスの信頼度スコアの閾値。デフォルトは0.50。

        Returns:
            np.ndarray: 指定されたクラスIDに属するオブジェクトのバウンディングボックスの配列。
        """
        pred_instances = mmdet_results.pred_instances

        class_results = []
        for box, label, score in zip(
            pred_instances.bboxes,
            pred_instances.labels,
            pred_instances.scores,
        ):
            if label == class_id and score > class_threshold:
                box = box.cpu().numpy()
                class_results.append(box)
            if score < class_threshold:
                break

        return np.array(class_results)
