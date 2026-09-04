import numpy as np


class YoloSegmentation:
    def __init__(self, weight_path, config):
        from ultralytics import YOLO
        self._model = YOLO(str(weight_path))
        if self._model.task != 'segment':
            raise ValueError(f'YOLO11-seg weight가 아닙니다: task={self._model.task}')
        self._config = config

    def target_mask(self, rgb, depth_mm):
        result = self._model.predict(rgb, conf=float(self._config['min_confidence']), verbose=False)[0]
        if result.boxes is None or result.masks is None or len(result.boxes) != 1:
            return None, 'TARGET_NOT_READY:EXPECTED_EXACTLY_ONE_MASK'
        mask = result.masks.data[0].cpu().numpy() > 0.5
        if mask.shape != rgb.shape[:2]:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        confidence = float(result.boxes.conf[0])
        valid = mask & np.isfinite(depth_mm) & (depth_mm > 0)
        if int(mask.sum()) < int(self._config['min_mask_pixels']):
            return None, 'TARGET_NOT_READY:MASK_TOO_SMALL'
        if valid.sum() / mask.sum() < float(self._config['min_depth_valid_ratio']):
            return None, 'TARGET_NOT_READY:DEPTH_QUALITY_LOW'
        return {'mask': mask, 'confidence': confidence, 'depth_valid_ratio': float(valid.sum() / mask.sum())}, ''
