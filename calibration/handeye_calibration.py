import os
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation

# 캘리브레이션 결과 저장 경로. 컨테이너(perception/grasp/control)가 볼륨 마운트로
# 읽어야 하므로 calibration/ 안이 아니라 data/ 아래에 둔다 (개발계획.md D-2).
HANDEYE_PATH = os.environ.get(
    "HANDEYE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "calibration", "T_gripper2camera.npy"),
)

def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    R = Rotation.from_euler('ZYZ', [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def find_checkerboard_pose(
    image, board_size, square_size, camera_matrix, dist_coeffs
):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * 25
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None

    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    retval, rvec, tvec = cv2.solvePnP(objp, corners_sub, camera_matrix, dist_coeffs)
    if not retval:
        return None, None

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec


def calibrate_camera_from_chessboard(
    image_folder_path,
    board_size,
    square_size,
):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size
    )

    obj_points = []
    img_points = []
    image_shape = None

    image_paths = image_folder_path

    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(corners_sub)

    if len(obj_points) < 1:
        print("체커보드 코너를 충분히 찾지 못하였습니다.")
        return None, None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_shape,
        None,
        None,
    )

    if not ret:
        print("캘리브레이션이 제대로 수렴하지 않았습니다.")
        return None, None, None, None

    return camera_matrix, dist_coeffs, rvecs, tvecs



if __name__ == "__main__":
    data = json.load(open("data/calibrate_data.json"))
    robot_poses = np.array(data["poses"])

    robot_poses[:, :3] = robot_poses[:, :3]
    image_paths = ["data/" + d for d in data["file_name"]]

    checkerboard_size = (10, 7)
    square_size = 25
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_chessboard(
        image_paths, checkerboard_size, square_size
    )

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_camera2checker_list = []
    t_camera2checker_list = []
    R_checker2camera_list = []
    t_checker2camera_list = []

    for img_path, pose in zip(image_paths, robot_poses):
        T_base2gripper = get_robot_pose_matrix(*pose)

        image = cv2.imread(img_path)
        if image is None:
            continue

        R_cam2checker, t_cam2checker = find_checkerboard_pose(
            image, checkerboard_size, square_size, camera_matrix, dist_coeffs
        )
        if R_cam2checker is None:
            continue

        T_gripper2base= T_base2gripper

        R_gripper2base = T_gripper2base[:3, :3]
        t_gripper2base = T_gripper2base[:3, 3]

        R_gripper2base_list.append(R_gripper2base.copy())
        t_gripper2base_list.append(t_gripper2base.reshape(-1, 1).copy())

        T_cam2checker = np.eye(4)
        T_cam2checker[:3, :3] = R_cam2checker
        T_cam2checker[:3, 3] = t_cam2checker.flatten()
        
        T_checker2cam = T_cam2checker

        R_checker2camera_list.append(T_checker2cam[:3, :3].copy())
        t_checker2camera_list.append(T_checker2cam[:3, 3].copy())


    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,
        t_gripper2base_list,
        R_checker2camera_list,
        t_checker2camera_list,
        method=cv2.CALIB_HAND_EYE_PARK,
    )


    T_base2gripper_example = get_robot_pose_matrix(*robot_poses[2])
    R_base2gripper_example = T_base2gripper_example[:3, :3]
    t_base2gripper_example = T_base2gripper_example[:3, 3]

    T_gripper2cam = np.eye(4)
    T_gripper2cam[:3, :3] = R_cam2gripper
    T_gripper2cam[:3, 3] = t_cam2gripper.flatten()

    T_base2cam = T_base2gripper_example @ T_gripper2cam

    print("===== Hand-Eye Calibration Results =====")
    print("R_base2gripper:\n", T_base2gripper_example[:3, :3])
    print("T_base2gripper:\n", T_base2gripper_example[:3, 3])
    print("\n")
    print("R_base2camera:\n", T_base2cam[:3, :3])
    print("T_base2camera:\n", T_base2cam[:3, 3])
    print("\n")
    print("R_gripper2camera:\n", T_gripper2cam[:3, :3])
    print("T_gripper2camera:\n", T_gripper2cam[:3, 3].tolist())

    os.makedirs(os.path.dirname(HANDEYE_PATH), exist_ok=True)
    np.save(HANDEYE_PATH, T_gripper2cam)
    print(f"saved to {HANDEYE_PATH}")
