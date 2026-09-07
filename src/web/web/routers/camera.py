"""카메라·뎁스 MJPEG 스트림 (화면정의서.md 2.2.4절 개정, 개발계획.md D-5: 1차 범위는 MJPEG).

executor(ros_bridge.RosExecutor)가 최근 프레임을 JPEG로 들고 있으면 그걸 무한
multipart 스트림으로 흘려보낸다. mock 모드나 카메라가 아직 프레임을 보내지 않았으면
executor가 None을 돌려주므로 204로 답한다 — <img> 태그가 깨진 아이콘 대신 onError로
빈 자리를 보여줄 수 있게(CameraViews.tsx 참조).

/api/camera/color는 원본이 아니라 grasp가 발행하는 /grasp/debug_image(검출 오버레이 +
파지 후보점)를 중계한다 — ros_bridge.py의 _BridgeNode 구독 참조. grasp가 아직 관측을
한 번도 처리하지 못했으면(예: 로봇 자세 미수신) 다른 스트림과 마찬가지로 204다.
"""
import asyncio

from fastapi import APIRouter, Request, Response
from starlette.responses import StreamingResponse

router = APIRouter()

BOUNDARY = "frame"
FPS = 10.0


async def _mjpeg(get_frame):
    interval = 1.0 / FPS
    while True:
        frame = get_frame()
        if frame is not None:
            yield (b"--" + BOUNDARY.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
        await asyncio.sleep(interval)


def _stream_or_204(get_frame):
    if get_frame() is None:
        return Response(status_code=204)
    return StreamingResponse(
        _mjpeg(get_frame),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


@router.get("/api/camera/color")
async def color_stream(request: Request):
    executor = request.app.state.executor
    return _stream_or_204(executor.latest_color_jpeg)


@router.get("/api/camera/depth")
async def depth_stream(request: Request):
    executor = request.app.state.executor
    return _stream_or_204(executor.latest_depth_jpeg)
