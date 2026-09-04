import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.utilities import remove_ros_args

from .live_scene_capture import LiveSceneCapture
from .model_runner import DISPLAY, METHODS, ModelRunner
from .pick_verifier import PickVerifier
from .result_writer import ResultWriter
from .robot_executor import RobotExecutor
from .yolo_segmentation import YoloSegmentation


def load_config(path):
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    required = ('assets', 'camera', 'robot', 'rg2', 'results_dir', 'model')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'설정 키 누락: {missing}')
    return config


def methods(value):
    if value == 'all':
        return METHODS
    selected = tuple(item.strip() for item in value.split(',') if item.strip())
    unknown = set(selected) - set(METHODS)
    if unknown:
        raise ValueError(f'지원하지 않는 method: {sorted(unknown)}')
    return selected


def blank(scene_id, round_id, trial_id, method, status, failure='', note='', model_log='', model_error_message=''):
    return {'scene_id': scene_id, 'round_id': round_id, 'trial_id': trial_id, 'method': DISPLAY[method],
            'status': status, 'candidate_count': 0, 'valid_width_count': 0, 'model_init_ms': '',
            'inference_ms': '', 'total_elapsed_ms': '', 'candidate_pose_camera': '', 'candidate_pose_robot': '',
            'transform_ok': False, 'ik_ok': False, 'rg2_grip_state': '', 'reobservation_ok': False,
             'pick_success': False, 'failure_code': failure, 'note': note, 'model_log': model_log,
             'model_error_message': model_error_message}


def main(argv=None):
    # launch_ros' Node action appends `--ros-args -r __node:=...` to argv; strip it before
    # argparse sees it, or the launch-driven run above dies with "unrecognized arguments".
    argv = remove_ros_args(args=sys.argv if argv is None else argv)[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--methods', default='all')
    parser.add_argument('--rounds', type=int, default=3); parser.add_argument('--input-mode', default='live')
    parser.add_argument('--reset-mode', default='manual'); parser.add_argument('--execute', default='false')
    args = parser.parse_args(argv)
    if args.input_mode != 'live' or args.reset_mode != 'manual' or args.rounds < 1:
        print('현재는 input_mode=live, reset_mode=manual, rounds>=1만 지원합니다.', file=sys.stderr)
        return 2
    execute = args.execute.lower() == 'true'
    config = load_config(args.config)
    selected = methods(args.methods)
    output = ResultWriter(Path(config['results_dir']).expanduser())
    rclpy.init()
    capture = LiveSceneCapture(config['camera'])
    executor = RobotExecutor(capture, {**config['robot'], **config['rg2']})
    # Early, one-shot diagnostic only — do NOT reuse this for the per-trial execute gate below.
    # get_current_posx has repeatedly been transiently slow in this cell; a single check here
    # can fail while the driver is briefly busy and then recover seconds later, which used to
    # lock the whole run into DRY_RUN_ONLY even though nothing was actually still wrong.
    ready, reason = executor.preflight(execute)
    if execute and not ready:
        print(f'실제 실행 차단(시작 시점 확인): {reason} — trial마다 다시 확인합니다.', file=sys.stderr)
    try:
        yolo = YoloSegmentation(Path(config['assets']['yolo_weight']).expanduser(), config['camera'])
    except Exception as exc:
        yolo = None
        yolo_error = f'YOLO_LOAD_FAILED:{type(exc).__name__}:{exc}'
    runner = ModelRunner(Path(config['assets']['checkpoints']).expanduser(), config['model']['max_inference_ms'])
    verifier = PickVerifier(capture, yolo) if yolo is not None else None
    scene_dir = output.directory / 'scenes'; scene_dir.mkdir(exist_ok=True)
    for round_id in range(1, args.rounds + 1):
        for method in selected:
            trial_id = f'r{round_id:02d}_{method}'
            started = time.monotonic()
            frame, capture_error = capture.capture()
            if frame is None:
                row = blank('', round_id, trial_id, method, 'TARGET_NOT_READY', capture_error)
            elif yolo is None:
                row = blank('', round_id, trial_id, method, 'ERROR', yolo_error)
            else:
                try:
                    target, target_error = yolo.target_mask(frame.rgb, frame.depth_mm)
                except Exception as exc:
                    target, target_error = None, f'YOLO_INFERENCE_FAILED:{type(exc).__name__}:{exc}'
                if target is None:
                    status = 'ERROR' if target_error.startswith('YOLO_INFERENCE_FAILED:') else 'TARGET_NOT_READY'
                    row = blank('', round_id, trial_id, method, status, target_error)
                else:
                    scene_id = f'live_{trial_id}'
                    scene_path = scene_dir / f'{scene_id}.npz'
                    runner.write_scene(frame, target, scene_path, scene_id)
                    try:
                        result, model_error, model_log = runner.run(method, scene_path, output.directory, trial_id)
                    except Exception as exc:
                        result, model_error, model_log = None, f'MODEL_RUNNER_FAILED:{type(exc).__name__}:{exc}', None
                    if result is None:
                        row = blank(scene_id, round_id, trial_id, method, 'ERROR', model_error,
                                    model_log=model_log.name if model_log else '')
                    else:
                        candidate = {'x_m': result.get('x_m'), 'y_m': result.get('y_m'), 'z_m': result.get('z_m')}
                        try:
                            width_ok, width_error = executor.validate_width(result)
                            if width_ok:
                                robot_pose, transform_ok, transform_error = executor.transform_and_validate(candidate)
                            else:
                                robot_pose, transform_ok, transform_error = None, False, width_error
                        except Exception as exc:
                            width_ok, robot_pose, transform_ok = False, None, False
                            transform_error = f'TRANSFORM_VALIDATION_FAILED:{type(exc).__name__}:{exc}'
                        model_message = str(result.get('error_message', ''))
                        row = blank(scene_id, round_id, trial_id, method, 'ERROR' if model_error else 'DRY_RUN_ONLY', model_error or transform_error,
                                    model_log=model_log.name if model_log else '', model_error_message=model_message)
                        row.update({'candidate_count': result.get('candidate_count', 0), 'valid_width_count': result.get('valid_width_count', 0),
                                    'model_init_ms': result.get('initialization_ms', ''), 'inference_ms': result.get('inference_ms', ''),
                                    'candidate_pose_camera': candidate, 'candidate_pose_robot': robot_pose or '', 'transform_ok': transform_ok})
                        if not model_error and width_ok and execute and transform_ok:
                            # Re-check right now, not the ready flag from process start — see
                            # the comment above the initial preflight() call.
                            trial_ready, trial_reason = executor.preflight(execute)
                            if not trial_ready:
                                print(f'실제 실행 차단({trial_id}): {trial_reason}', file=sys.stderr)
                                row['failure_code'] = trial_reason
                            else:
                                try:
                                    picked, code = executor.execute_pick(robot_pose)
                                    grip_state, reobserved, verify_code = verifier.verify(executor.last_grip, target)
                                except Exception as exc:
                                    picked, grip_state, reobserved = False, '', False
                                    code, verify_code = f'EXECUTION_EXCEPTION:{type(exc).__name__}:{exc}', ''
                                # A failed trial may leave the arm below Home. Always attempt the
                                # configured return path before asking the operator for the next reset.
                                try:
                                    home_ok = executor.return_home()
                                except Exception as exc:
                                    home_ok = False
                                    code = code or f'HOME_RETURN_EXCEPTION:{type(exc).__name__}:{exc}'
                                pick_success = picked and grip_state == 'gripped' and reobserved
                                row.update({'status': 'OK' if pick_success and home_ok else 'EXECUTION_FAILED', 'ik_ok': False,
                                            'rg2_grip_state': grip_state, 'reobservation_ok': reobserved, 'pick_success': pick_success,
                                            'failure_code': code or verify_code or ('' if home_ok else 'HOME_RETURN_FAILED')})
            row['total_elapsed_ms'] = round((time.monotonic() - started) * 1000, 2)
            output.add(row)
            output.save()
            print(f'{trial_id}: {row["status"]} {row["failure_code"]}')
            if args.reset_mode == 'manual' and sys.stdin.isatty() and not (round_id == args.rounds and method == selected[-1]):
                input('물체를 작업영역에 다시 놓고 로봇이 Home에 있는지 확인한 뒤 Enter를 누르세요: ')
    capture.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
