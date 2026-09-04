"""FastAPI 진입점. ROS2를 모르는 순수 HTTP 계층으로 유지 (시스템명세서 3.1절).

**이 파일과 routers/ 하위는 rclpy를 import하지 않는다.** 로봇에 닿는 경로는 전부
`executor.py` 어댑터를 지나며, ROS2를 아는 코드는 `ros_bridge.py` 하나뿐이다.
"""
import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import executor as executor_mod
from .events import hub
from .routers import camera, commands, confirmations, executions, live, robot, traces, world

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
app.include_router(world.router)
app.include_router(live.router)
app.include_router(camera.router)


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


# --- 프론트엔드 정적 자산 ----------------------------------------------------
# 라우터를 전부 등록한 뒤에 마운트한다 — SPA 폴백이 /api·/ws를 가로채면 안 된다.
FRONTEND_DIST = pathlib.Path(os.environ.get("FRONTEND_DIST", "/app/frontend_dist"))

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        """클라이언트 라우팅(/, /history)을 위한 폴백.

        실제 파일이 있으면 그 파일을, 없으면 index.html을 돌려준다. 브라우저에서
        /history를 직접 새로고침해도 404가 나지 않게 하는 부분이다.
        """
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    logger.warning("프론트엔드 빌드 산출물이 없습니다: %s (API만 제공)", FRONTEND_DIST)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
