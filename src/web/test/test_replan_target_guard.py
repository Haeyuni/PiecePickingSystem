"""재계획이 명령에 없던 물체를 대상으로 삼는 것을 막는다 (2026-09-08).

실물에서 나온 사고 그대로다. "물티슈 왼쪽으로 옮겨줘"가 파지 실패한 뒤 재계획을 받은
planner LLM이

    refusal='물티슈는 파지 실패로 인해 건너뛰고, 접이 우산만 옮깁니다.'

라며 **사용자가 말한 적 없는 접이 우산**을 집으러 갔다. `previous_failure`를 알려주면
LLM이 "실패한 건 건너뛰고 대신 다른 걸 하자"로 읽는 것인데, 그건 명령의 범위를 벗어난
판단이다 — 못 옮기는 것이지 다른 물체를 대신 옮겨도 되는 게 아니다.

`_run_command_body`를 통째로 돌리려면 planner/executor/perception이 다 필요하므로,
여기서는 그 함수가 쓰는 **판정 규칙만** 떼어 검증한다. 규칙 자체는 세 줄이고, 지켜야 할
성질이 분명하다:

  - 처음 계획에 없던 클래스가 들어오면 막는다
  - 실패한 물체를 빼고 나머지를 계속하는 것(부분집합)은 허용한다
  - object_id는 관측마다 바뀌므로 **id가 아니라 클래스**로 본다
"""
import ast
import pathlib
import unittest

SOURCE = pathlib.Path(__file__).parents[1] / "web" / "orchestrator.py"


def _function(name, namespace=None):
    """orchestrator에서 함수 하나만 떼어 컴파일한다 — FastAPI/DB/planner를 import하지 않는다."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class_of = _function("class_of")
# 실제 프로덕션 함수를 그대로 떼어 쓴다 — 판정 로직을 테스트용으로 다시 베끼면
# 코드와 테스트가 각자 따로 고쳐질 수 있다. class_of를 의존하므로 같이 넣어준다.
_off_scope_intruders = _function("_off_scope_intruders", {"class_of": class_of})


def intruders(allowed_classes, targets):
    """`_off_scope_intruders`와 같은 판정을 (skill, id, class) 튜플 목록에 바로 적용한다.

    world_state를 만들지 않고 이미 알고 있는 클래스명으로 바로 시험하고 싶은 기존 테스트들을
    위한 얇은 래퍼다 — 실제 판정은 여전히 `_off_scope_intruders`가 한다.
    """
    world = {"objects": [{"object_id": oid, "class_name": name}
                         for _, oid, name in targets]}
    steps = [{"skill": skill, "object_id": oid} for skill, oid, _ in targets]
    return _off_scope_intruders(steps, world, allowed_classes)


class ClassOfTest(unittest.TestCase):

    def setUp(self):
        self.world = {"objects": [
            {"object_id": "obj_014", "class_name": "wet_wipes"},
            {"object_id": "obj_013", "class_name": "umbrella"},
        ]}

    def test_looks_up_class(self):
        self.assertEqual(class_of(self.world, "obj_014"), "wet_wipes")
        self.assertEqual(class_of(self.world, "obj_013"), "umbrella")

    def test_unknown_object(self):
        self.assertIsNone(class_of(self.world, "obj_999"))

    def test_empty_world(self):
        self.assertIsNone(class_of(None, "obj_014"))
        self.assertIsNone(class_of({}, "obj_014"))


class ReplanTargetGuardTest(unittest.TestCase):

    def test_the_real_umbrella_case_is_blocked(self):
        """2026-09-08 실물: 물티슈 → 우산 치환."""
        allowed = {"wet_wipes"}
        targets = [("pick", "obj_013", "umbrella"), ("place_into", "obj_013", "umbrella")]
        self.assertEqual(intruders(allowed, targets), {"umbrella"})

    def test_same_object_with_new_id_is_allowed(self):
        """object_id는 관측마다 바뀐다 — id로 비교하면 정상 재시도가 전부 막힌다."""
        allowed = {"wet_wipes"}
        targets = [("pick", "obj_027", "wet_wipes"), ("place_into", "obj_027", "wet_wipes")]
        self.assertEqual(intruders(allowed, targets), set())

    def test_skipping_a_failed_object_is_allowed(self):
        """여러 물체 명령에서 실패한 하나를 빼고 계속하는 것은 범위 안이다."""
        allowed = {"wet_wipes", "toothpaste", "sunscreen"}
        targets = [("pick", "obj_002", "toothpaste"), ("place_into", "obj_002", "toothpaste"),
                   ("pick", "obj_003", "sunscreen"), ("place_into", "obj_003", "sunscreen")]
        self.assertEqual(intruders(allowed, targets), set())

    def test_place_only_steps_do_not_widen_the_allowed_set(self):
        """판정은 pick 대상만 본다 — 로봇이 실제로 집는 것은 pick 스텝이다."""
        allowed = {"wet_wipes"}
        targets = [("place_into", "obj_013", "umbrella")]
        self.assertEqual(intruders(allowed, targets), set())

    def test_unknown_class_is_treated_as_new(self):
        """판별할 수 없으면 멈추는 쪽이 엉뚱한 물체를 집는 것보다 낫다."""
        allowed = {"wet_wipes"}
        targets = [("pick", "obj_030", None)]
        self.assertEqual(intruders(allowed, targets), {None})

    def test_first_attempt_defines_the_allowed_set(self):
        allowed = None
        targets = [("pick", "obj_014", "wet_wipes"), ("place_into", "obj_014", "wet_wipes")]
        step_classes = {name for skill, _, name in targets if skill == "pick"}
        if allowed is None:
            allowed = step_classes
        self.assertEqual(allowed, {"wet_wipes"})
        self.assertEqual(intruders(allowed, targets), set())


if __name__ == "__main__":
    unittest.main()
