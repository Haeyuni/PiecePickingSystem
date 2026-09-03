"""FastAPI 진입점. ROS2를 모르는 순수 HTTP 계층으로 유지 (시스템명세서 3.1절).

trace_id 발급, 시퀀스 미리보기, 실행 로그 조회 등은 routers/ 하위로 분리한다.
"""
from fastapi import FastAPI

app = FastAPI(title="piece-picking-system web")


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
