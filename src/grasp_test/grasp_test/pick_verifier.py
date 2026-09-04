class PickVerifier:
    """물리적 파지 확인. 두 가지를 모두 통과해야 pick_success다:

    1) RG2 상태(robot_executor._gripper_close가 이미 읽어온 stalled/reached_goal/width) —
       그리퍼가 힘으로 멈췄는지만 보면 안 된다. 다른 이유로 멈출 수도 있어서다.
    2) 재관측 — 들어올린 뒤 원래 자리를 다시 찍어 같은 클래스가 더 이상 보이지 않는지 확인한다.
       그리퍼 폭만으로는 "쥐긴 했는데 다른 물체를 스쳐서 폭이 남았다" 같은 헛파지를 못 걸러낸다.

    카메라 재촬영만 하고 로봇을 움직이지 않으므로 안전하다.
    """
    def __init__(self, capture, yolo):
        self._capture = capture
        self._yolo = yolo

    def verify(self, grip):
        if grip is None:
            return '', False, 'RG2_STATUS_UNAVAILABLE'
        grip_state = grip['grip_state']
        if grip_state != 'gripped':
            return grip_state, False, 'RG2_EMPTY'

        frame, capture_error = self._capture.capture()
        if frame is None:
            return grip_state, False, f'REOBSERVATION_CAPTURE_FAILED:{capture_error}'
        target, _ = self._yolo.target_mask(frame.rgb, frame.depth_mm)
        if target is not None:
            return grip_state, False, 'OBJECT_STILL_ON_TABLE'
        return grip_state, True, ''
