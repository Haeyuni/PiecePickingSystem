"""FastAPI 진입점. ROS2를 모르는 순수 HTTP 계층으로 유지 (시스템명세서 3.1절).

**이 파일과 routers/ 하위는 rclpy를 import하지 않는다.** 로봇에 닿는 경로는 전부
`executor.py` 어댑터를 지나며, ROS2를 아는 코드는 `ros_bridge.py` 하나뿐이다.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import executor as executor_mod
from .events import hub
from .routers import commands, confirmations, executions, live, robot, traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor = executor_mod.create_executor()
    await executor.start()

    async def on_state_event(event: dict) -> None:
        # robot_state / safety_event를 그대로 브라우저로 중계한다(4절)
        await hub.broadcast(event)

    executor.subscribe_state(on_state_event)
    app.state.executor = executor
    logger.info("web 기동 완료 (mock=%s)", executor_mod.is_mock())
    try:
        yield
    finally:
        await executor.close()


app = FastAPI(title="piece-picking-system web", lifespan=lifespan)

app.include_router(commands.router)
app.include_router(traces.router)
app.include_router(executions.router)
app.include_router(confirmations.router)
app.include_router(robot.router)
app.include_router(live.router)


@app.get("/health")
def health():
    executor = app.state.executor
    world_state = executor.get_latest_world_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "mock_mode": executor_mod.is_mock(),
        "robot_state": executor.robot_state(),
        "world_state_objects": len(world_state.get("objects", [])) if world_state else None,
        "ws_clients": hub.client_count,
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
