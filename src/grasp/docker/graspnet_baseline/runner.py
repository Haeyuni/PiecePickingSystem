"""GraspNet-baseline 추론 런타임.

두 가지 모드로 돈다.

  1) 1회용 파일 모드 (원래 방식):
         python runner.py <input.npz> <output.json> <checkpoint> <device> \
                          <num_points> <min_width_m> <max_width_m> <max_candidates>
     호출 한 번에 컨테이너가 뜨고 죽는다. 매번 torch/CUDA import + 모델 생성 +
     checkpoint 로드를 다시 하므로 **추론 1회가 수십~수백 초**다(2026-09-05 실물에서
     60초 타임아웃이 연속으로 초과돼 180초로 올려야 했다). grasp 노드가 호스트에서
     네이티브로 돌 때의 fallback 경로로 남겨둔다.

  2) 상주 서버 모드 (2026-09-06 추가, 권장):
         python runner.py --serve <checkpoint> <device> <port>
     기동할 때 모델을 **한 번만** 올리고 HTTP로 추론 요청을 받는다. 콜드 스타트를
     기동 1회로 밀어내는 것이 목적이다. 이러면 grasp 노드가 compose 컨테이너 안에
     그대로 있어도 된다 — 1)은 grasp 노드가 `docker run`을 직접 불러야 해서 Docker
     socket/CLI가 필요했고(그래서 호스트 네이티브 실행이 강제됐다), 컨테이너 안 임시
     디렉터리 경로를 호스트 데몬이 해석하는 문제도 있었다.

     엔드포인트:
       GET  /health          → 200 {"status": "ready"}
       POST /infer?<params>  → body: .npy (float32 Nx3, camera frame, meter)
                               200 {"input_frame","input_unit","candidates"}
                               409 이미 다른 추론이 진행 중 (호출부가 이번 관측을 건너뛴다)

  3) python runner.py --check-checkpoint <checkpoint>   (setup 스크립트용)

**HTTP 서버는 stdlib만 쓴다.** 이 이미지는 python 3.8 conda 환경에 graspnetAPI/open3d/
psutil 버전을 아슬아슬하게 맞춰 놓은 상태다(Dockerfile 주석의 사고 기록 참조) — fastapi
같은 걸 추가로 깔면 그 조합이 깨질 위험이 실제로 있다.
"""
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

ROOT = Path('/opt/graspnet-baseline')
for item in (ROOT / 'models', ROOT / 'dataset', ROOT / 'utils'):
    sys.path.insert(0, str(item))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup

# GPU는 하나뿐이다. 추론 두 개가 겹치면 메모리가 바닥난다 — 실물에서 컨테이너가 20개 넘게
# 쌓여 perception까지 OOM으로 죽은 적이 있다(graspnet_baseline.py 모듈 docstring). 서버
# 모드는 ThreadingHTTPServer라 요청이 동시에 들어올 수 있으므로 여기서 직렬화한다.
_INFERENCE_LOCK = threading.Lock()


def load_model(checkpoint_path, device):
    """GraspNet 모델 + checkpoint를 올린다. 서버 모드에서는 기동 시 한 번만 부른다."""
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=.05, hmin=-.02, hmax_list=[.01, .02, .03, .04],
                   is_training=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state['model_state_dict'])
    net.eval()
    return net


def infer(net, device, points, num_points, min_width_m, max_width_m, max_candidates):
    """camera frame meter point cloud → 후보 리스트. 두 모드가 공유하는 본체."""
    if len(points) == 0:
        raise RuntimeError('입력 point cloud가 비어 있습니다')
    if not np.isfinite(points).all():
        raise RuntimeError('입력 points_cam_m에 유한하지 않은 값이 있습니다')
    num_points = int(num_points)
    print(f'GraspNet input: camera frame, meter, points={len(points)}, '
          f'z=[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]', flush=True)
    rng = np.random.default_rng(0)
    indexes = rng.choice(len(points), num_points, replace=len(points) < num_points)
    input_points = torch.from_numpy(points[indexes][None].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = pred_decode(net({
            'point_clouds': input_points,
            'cloud_colors': np.zeros((num_points, 3), dtype=np.float32),
        }))[0].detach().cpu().numpy()
    group = GraspGroup(pred)
    group = group[(group.widths >= float(min_width_m)) & (group.widths <= float(max_width_m))]
    group.nms()
    group.sort_by_score()
    # depth를 반드시 같이 내보낸다. GraspNet의 translation은 손끝이 아니라 그리퍼
    # 기준점이고, 실제로 무는 지점은 **translation + depth x 접근축**이다
    # (graspnetAPI utils.plot_gripper_pro_max: 손가락 상자가 center에서 +X로 depth만큼
    # 뻗는다). 이 값을 버리면 파지점이 접근축 방향으로 10~40mm 앞에 찍혀 허공에서 닫힌다.
    return [{
        'score': float(grasp.score),
        'width_m': float(grasp.width),
        'depth_m': float(grasp.depth),
        'rotation_matrix': grasp.rotation_matrix.tolist(),
        'translation_m': grasp.translation.tolist(),
    } for grasp in group[:int(max_candidates)]]


def main(input_path, output_path, checkpoint_path, device_name, num_points, min_width_m,
         max_width_m, max_candidates):
    """1회용 파일 모드."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU를 찾지 못했습니다')
    device = torch.device(device_name)
    points = np.load(input_path, allow_pickle=False)['points_cam_m']
    net = load_model(checkpoint_path, device)
    candidates = infer(net, device, points, num_points, min_width_m, max_width_m,
                       max_candidates)
    Path(output_path).write_text(json.dumps({
        'input_frame': 'camera', 'input_unit': 'm', 'candidates': candidates,
    }), encoding='utf-8')


class _Handler(BaseHTTPRequestHandler):
    # 클래스 속성으로 주입한다 — BaseHTTPRequestHandler는 요청마다 인스턴스를 새로 만든다.
    net = None
    device = None

    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        # 기본 구현은 stderr로 매 요청을 찍는다. 추론 로그(infer의 print)만 남기면 충분하다.
        pass

    def _reply(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != '/health':
            self._reply(404, {'error': 'not found'})
            return
        self._reply(200, {'status': 'ready'})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/infer':
            self._reply(404, {'error': 'not found'})
            return
        query = parse_qs(parsed.query)

        def arg(name, default):
            return query.get(name, [default])[0]

        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else b''
        if not body:
            self._reply(400, {'error': 'empty body'})
            return

        # 이미 도는 추론이 있으면 기다리지 않고 409로 돌려준다. 호출부(grasp 노드)는
        # 관측 주기가 추론보다 빠를 때 이번 관측을 그냥 건너뛰도록 만들어져 있다 —
        # 큐에 쌓으면 낡은 point cloud로 뒤늦게 추론하는 셈이라 의미가 없다.
        if not _INFERENCE_LOCK.acquire(blocking=False):
            self._reply(409, {'error': 'inference busy'})
            return
        try:
            points = np.load(io.BytesIO(body), allow_pickle=False)
            points = np.asarray(points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                self._reply(400, {'error': f'expected Nx3 points, got {points.shape}'})
                return
            candidates = infer(self.net, self.device, points,
                               arg('num_points', '20000'), arg('min_width_m', '0.005'),
                               arg('max_width_m', '0.110'), arg('max_candidates', '5'))
        except Exception as exc:  # 서버는 요청 하나 때문에 죽으면 안 된다
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            self._reply(500, {'error': f'{type(exc).__name__}: {exc}'})
            return
        finally:
            _INFERENCE_LOCK.release()
        self._reply(200, {'input_frame': 'camera', 'input_unit': 'm',
                          'candidates': candidates})


def serve(checkpoint_path, device_name, port):
    """상주 서버 모드. 모델을 먼저 올리고 나서 포트를 연다 —
    /health가 200이면 곧바로 추론을 받을 수 있다는 뜻이어야 하기 때문이다."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU를 찾지 못했습니다')
    device = torch.device(device_name)
    print(f'GraspNet 모델 로딩: {checkpoint_path} → {device_name}', flush=True)
    _Handler.net = load_model(checkpoint_path, device)
    _Handler.device = device
    print(f'GraspNet 준비 완료 — 0.0.0.0:{port} 대기', flush=True)
    ThreadingHTTPServer(('0.0.0.0', int(port)), _Handler).serve_forever()


if __name__ == '__main__':
    try:
        if sys.argv[1] == '--check-checkpoint':
            state = torch.load(sys.argv[2], map_location='cpu')
            if 'model_state_dict' not in state:
                raise RuntimeError('model_state_dict가 없습니다')
            print('checkpoint.tar 확인 완료')
        elif sys.argv[1] == '--serve':
            serve(*sys.argv[2:])
        else:
            main(*sys.argv[1:])
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
