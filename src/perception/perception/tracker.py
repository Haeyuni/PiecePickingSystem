"""프레임 사이에서 object_id를 유지한다.

**왜 필요한가**: `object_id`는 계획과 실행을 잇는 이름이다. 재계획(FR-16)은 실패한 물체의
`object_id`를 새 `world_state`에서 다시 찾아 건너뛰는데, 프레임마다 번호를 새로 매기면
그 이름이 다른 물체를 가리키게 된다 — 실패한 물체를 다시 집고, 멀쩡한 물체를 건너뛴다.

**어떻게**: 클래스가 같고 3D 거리가 임계값 안이면 같은 물체로 본다. 물체가 대체로 정지해
있는 1차 범위(BR 4.1절 1~5단계)에는 이것으로 충분하다. 컨베이어 위 이동 물체(10단계
확장)에는 부족하고, 그때는 속도 모델이 있는 추적기로 교체해야 한다.
"""
import itertools


class ObjectTracker:
    def __init__(self, match_distance_mm: float = 60.0, forget_after_misses: int = 5):
        self._match_distance_mm = match_distance_mm
        self._forget_after_misses = forget_after_misses
        self._counter = itertools.count(1)
        # object_id → {"class_name", "position", "misses"}
        self._tracks: dict[str, dict] = {}

    def assign(self, detections: list[dict]) -> list[str]:
        """검출 목록에 object_id를 붙인다. detections[i]는 class_name·position을 가진다.

        position은 base 좌표계 (x, y, z) mm이거나 None(좌표를 못 낸 경우)이다. 좌표가 없으면
        위치로 이을 수 없으므로 새 id를 준다 — 잘못 이어 붙이는 것보다 새 물체로 보는 편이 안전하다.
        """
        unmatched = dict(self._tracks)
        assigned: list[str | None] = [None] * len(detections)

        # 가까운 짝부터 확정한다. 먼 것부터 붙이면 더 가까운 후보를 빼앗을 수 있다.
        pairs = []
        for index, detection in enumerate(detections):
            position = detection.get("position")
            if position is None:
                continue
            for object_id, track in unmatched.items():
                if track["class_name"] != detection["class_name"]:
                    continue
                distance = _distance(position, track["position"])
                if distance <= self._match_distance_mm:
                    pairs.append((distance, index, object_id))
        pairs.sort()

        taken_ids: set[str] = set()
        for _, index, object_id in pairs:
            if assigned[index] is not None or object_id in taken_ids:
                continue
            assigned[index] = object_id
            taken_ids.add(object_id)

        for index, detection in enumerate(detections):
            object_id = assigned[index]
            if object_id is None:
                object_id = f"obj_{next(self._counter):03d}"
                assigned[index] = object_id
            self._tracks[object_id] = {
                "class_name": detection["class_name"],
                "position": detection.get("position") or self._tracks.get(object_id, {}).get("position"),
                "misses": 0,
            }

        # 이번 프레임에 안 보인 트랙은 몇 프레임 더 들고 있다가 버린다.
        # 한 프레임 가려졌다고 잊으면 다시 보일 때 새 id가 붙어 추적이 끊긴다.
        for object_id in list(self._tracks):
            if object_id in taken_ids or object_id in assigned:
                continue
            self._tracks[object_id]["misses"] += 1
            if self._tracks[object_id]["misses"] > self._forget_after_misses:
                del self._tracks[object_id]

        return [object_id for object_id in assigned]


def _distance(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return sum((float(p) - float(q)) ** 2 for p, q in zip(a, b)) ** 0.5
