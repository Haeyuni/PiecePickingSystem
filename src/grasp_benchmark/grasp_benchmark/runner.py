import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


METHODS = ('pca_normal', 'ggcnn', 'graspnet_baseline', 'contact_graspnet')
DISPLAY_NAMES = {
    'pca_normal': 'PCA_Normal',
    'ggcnn': 'GG-CNN',
    'graspnet_baseline': 'GraspNet_baseline',
    'contact_graspnet': 'Contact_GraspNet',
}


def resolve_scene(value):
    requested = Path(value).expanduser()
    choices = (requested, Path.cwd() / 'data' / 'scenes' / requested.name, Path.home() / 'Downloads' / requested.name)
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'scene 파일을 찾지 못했습니다: {value}')


def error_row(scene_id, method, message):
    return {
        'scene_id': scene_id, 'method': DISPLAY_NAMES[method], 'status': 'ERROR',
        'candidate_count': 0, 'valid_width_count': 0, 'best_score': None,
        'initialization_ms': None, 'inference_ms': None, 'width_m': None,
        'x_m': None, 'y_m': None, 'z_m': None, 'u_px': None, 'v_px': None,
        'width_validation': 'NOT_CHECKED', 'candidate_definition': '',
        'note': '', 'error_message': message,
        'tested_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def run_logged(command, log_path):
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'$ {shlex_join(command)}\n')
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def shlex_join(command):
    try:
        import shlex
        return shlex.join(command)
    except AttributeError:
        return ' '.join(command)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run four grasp models in isolated Docker containers.')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args(argv)

    try:
        scene_path = resolve_scene(args.scene)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if shutil.which('docker') is None:
        print('docker 명령을 찾지 못했습니다.', file=sys.stderr)
        return 2

    package_share = Path(get_package_share_directory('grasp_benchmark'))
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / '.models'
    models_dir.mkdir(exist_ok=True)
    scene_id = scene_path.stem
    for method in METHODS:
        (results_dir / f'{scene_id}_{method}.json').unlink(missing_ok=True)
        (results_dir / f'{scene_id}_{method}.log').unlink(missing_ok=True)
    for suffix in ('comparison.csv', 'comparison.xlsx', 'comparison_preview.png'):
        (results_dir / f'{scene_id}_{suffix}').unlink(missing_ok=True)

    for method in METHODS:
        image = f'piece-picking-grasp-benchmark-{method}:latest'
        log_path = results_dir / f'{scene_id}_{method}.log'
        dockerfile = package_share / 'docker' / method / 'Dockerfile'
        build = ['docker', 'build', '--tag', image, '--file', str(dockerfile), str(package_share)]
        print(f'===== {DISPLAY_NAMES[method]} build/run =====')
        if run_logged(build, log_path) != 0:
            (results_dir / f'{scene_id}_{method}.json').write_text(json.dumps(error_row(scene_id, method, 'Docker image build failed'), ensure_ascii=False, indent=2))
            continue
        run = ['docker', 'run', '--rm', '-v', f'{scene_path.parent}:/scenes:ro', '-v', f'{results_dir}:/results', '-v', f'{models_dir}:/models']
        if method != 'pca_normal':
            run.extend(['--gpus', 'all'])
        run.extend([image, f'/scenes/{scene_path.name}'])
        if run_logged(run, log_path) != 0 and not (results_dir / f'{scene_id}_{method}.json').exists():
            (results_dir / f'{scene_id}_{method}.json').write_text(json.dumps(error_row(scene_id, method, 'Container exited before writing a result JSON'), ensure_ascii=False, indent=2))

    aggregate_log = results_dir / f'{scene_id}_aggregate.log'
    aggregate_image = 'piece-picking-grasp-benchmark-aggregate:latest'
    aggregate_dockerfile = package_share / 'docker' / 'aggregate' / 'Dockerfile'
    build = ['docker', 'build', '--tag', aggregate_image, '--file', str(aggregate_dockerfile), str(package_share)]
    if run_logged(build, aggregate_log) == 0:
        run = ['docker', 'run', '--rm', '-v', f'{scene_path.parent}:/scenes:ro', '-v', f'{results_dir}:/results', aggregate_image, f'/scenes/{scene_path.name}']
        run_logged(run, aggregate_log)
    else:
        print(f'집계 이미지 빌드 실패: {aggregate_log}', file=sys.stderr)
        return 1
    print(f'완료: {results_dir / (scene_id + "_comparison.xlsx")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
