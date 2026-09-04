import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml

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


def blank(scene_id, round_id, trial_id, method, status, failure='', note=''):
    return {'scene_id': scene_id, 'round_id': round_id, 'trial_id': trial_id, 'method': DISPLAY[method],
            'status': status, 'candidate_count': 0, 'valid_width_count': 0, 'model_init_ms': '',
            'inference_ms': '', 'total_elapsed_ms': '', 'candidate_pose_camera': '', 'candidate_pose_robot': '',
            'transform_ok': False, 'ik_ok': False, 'rg2_grip_state': '', 'reobservation_ok': False,
            'pick_success': False, 'failure_code': failure, 'note': note}


def main(argv=None):
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
    executor = RobotExecutor(capture, {**config['robot'], 'rg2_command_action': config['rg2'].get('command_action'),
                                        'rg2_state_topic': config['rg2'].get('state_topic')})
    ready, reason = executor.preflight(execute)
    if execute and not ready:
        print(f'실제 실행 차단: {reason}', file=sys.stderr)
    try:
        yolo = YoloSegmentation(Path(config['assets']['yolo_weight']).expanduser(), config['camera'])
    except Exception as exc:
        yolo = None
        yolo_error = f'YOLO_LOAD_FAILED:{type(exc).__name__}:{exc}'
    runner = ModelRunner(Path(config['assets']['checkpoints']).expanduser(), config['model']['max_inference_ms'])
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
                target, target_error = yolo.target_mask(frame.rgb, frame.depth_mm)
                if target is None:
                    row = blank('', round_id, trial_id, method, 'TARGET_NOT_READY', target_error)
                else:
                    scene_id = f'live_{trial_id}'
                    scene_path = scene_dir / f'{scene_id}.npz'
                    runner.write_scene(frame, target, scene_path, scene_id)
                    result, model_error, _ = runner.run(method, scene_path, output.directory, trial_id)
                    if result is None:
                        row = blank(scene_id, round_id, trial_id, method, 'ERROR', model_error)
                    else:
                        candidate = {'x_m': result.get('x_m'), 'y_m': result.get('y_m'), 'z_m': result.get('z_m')}
                        robot_pose, transform_ok, transform_error = executor.transform_and_validate(candidate)
                        row = blank(scene_id, round_id, trial_id, method, 'ERROR' if model_error else 'DRY_RUN_ONLY', model_error or transform_error)
                        row.update({'candidate_count': result.get('candidate_count', 0), 'valid_width_count': result.get('valid_width_count', 0),
                                    'model_init_ms': result.get('initialization_ms', ''), 'inference_ms': result.get('inference_ms', ''),
                                    'candidate_pose_camera': candidate, 'candidate_pose_robot': robot_pose or '', 'transform_ok': transform_ok})
                        if not model_error and execute and transform_ok and ready:
                            picked, code = executor.execute_pick(robot_pose)
                            grip, reobserved, verify_code = PickVerifier().verify()
                            row.update({'status': 'OK' if picked and grip and reobserved else 'EXECUTION_FAILED',
                                        'rg2_grip_state': grip, 'reobservation_ok': reobserved, 'pick_success': picked and grip and reobserved,
                                        'failure_code': code or verify_code})
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
