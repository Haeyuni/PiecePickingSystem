# mock 월드 상태 픽스처

`perception`/`grasp` 없이 planner·web을 개발·검증하기 위한 `WorldState` 고정 데이터
(개발계획.md 4.2절 B0).

**구조는 `sort_msgs/msg/WorldState.msg`를 JSON으로 직렬화한 것과 동일하다.** 필드를
추가·변경하면 이 파일들도 함께 고쳐야 한다 — 안 그러면 mock으로 통과한 코드가 실물에서 깨진다.

| 파일 | 시나리오 | 이 픽스처로 확인하는 것 |
|---|---|---|
| `world_state_normal.json` | 정상 다물체 (3종) | 일반 분류 명령이 유효한 시퀀스로 계획되는가 |
| `world_state_unknown_class.json` | 신규 클래스 포함 | `needs_confirmation=true` 물체가 `fallback`(fragile) 프로파일로 강제되는가 (FR-05b, NFR-03a). 이 물체는 일부러 `profile: normal`·`fragile: false`로 두었다 — VLM 제안값을 그대로 쓰면 테스트가 통과해버려 강제 여부를 증명하지 못한다 |
| `world_state_missing_target.json` | 지시 대상이 없음 | 미검출 물체 지시가 시퀀스 대신 거부로 나오는가 (FR-11) |
| `world_state_not_graspable.json` | `graspable=false` 포함 | 파지 불가 물체가 계획에서 제외되고 `needs_reobserve`에 담기는가 (FR-03) |
| `world_state_empty.json` | 빈 목록 | 물체가 없을 때 거부 응답과 빈 상태 UI가 동작하는가 |
| `world_state_overweight.json` | 가반하중 초과 물체 포함 (7200g) | 검증기가 가반하중(5000g)을 넘는 물체를 계획에서 막는가 (B5 평가셋). 파지 가능한 캔을 함께 둔 이유는 "무거운 것 하나 때문에 전체가 막히는지"를 같이 보기 위해서다 |
| `world_state_out_of_workspace.json` | 작업반경 밖 물체 포함 (958mm) | 검증기가 작업반경(900mm) 밖 파지 위치를 막는가 (B5 평가셋). **LLM은 좌표를 보지 못하므로**(`grounding.describe_for_prompt`가 좌표를 넣지 않는다) 이 케이스는 프롬프트가 아니라 검증기를 시험한다 |

단위는 mm (인터페이스_정의서.md 1.1절). 좌표는 M0609 작업반경(900mm) 안의 임의값이다 — `world_state_out_of_workspace.json`의 `obj_007`만 예외이고, 그 픽스처는 반경을 넘기는 것이 목적이다.
