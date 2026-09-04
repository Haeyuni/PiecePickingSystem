import sys
import time

import numpy as np

from common.scene_io import RG2_MAX_WIDTH_M, RG2_MIN_WIDTH_M, load_scene, result, write_result


def main(path):
    scene = load_scene(path)
    start = time.perf_counter()
    points = scene['target_points']
    center = np.median(points, axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - center), rowvar=False))
    closing = vectors[:, np.argsort(values)[::-1]][:, 1]
    closing /= np.linalg.norm(closing)
    width = float(np.percentile((points - center) @ closing, 95) - np.percentile((points - center) @ closing, 5))
    valid = int(RG2_MIN_WIDTH_M <= width <= RG2_MAX_WIDTH_M)
    write_result(result(scene, 'PCA_Normal', 'OK' if valid else 'WIDTH_REJECTED', candidate_count=1,
                        valid_width_count=valid, inference_ms=round((time.perf_counter() - start) * 1000, 2),
                        width_m=width, x_m=float(center[0]), y_m=float(center[1]), z_m=float(center[2]),
                        width_validation='VALID' if valid else 'REJECTED', candidate_definition='PCA representative grasp: 1',
                        note='기하학 기준선; 학습 score 없음'))


if __name__ == '__main__':
    main(sys.argv[1])
