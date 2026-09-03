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

단위는 mm (인터페이스_정의서.md 1.1절). 좌표는 M0609 작업반경(900mm) 안의 임의값이다.
