import { useEffect, useState } from 'react'

/** 관측 뷰 (화면정의서.md 2.2.4절 개정, D-5)
 *
 * MJPEG(개발계획 D-5 1차 범위)를 <img>에 그대로 물린다 — 브라우저가
 * multipart/x-mixed-replace를 네이티브로 처리하므로 별도 스트리밍 클라이언트가
 * 필요 없다. mock 모드나 카메라 미연결 시 백엔드가 204를 돌려주는데, <img>는 204를
 * 에러로 보고 onError를 발화한다 — 그때만 안내 문구로 대체한다(깨진 아이콘 대신).
 *
 * "카메라 뷰"는 원본이 아니라 grasp가 합성한 /grasp/debug_image다 — 캡처 사진 위에
 * 마스크 윤곽·신뢰도(perception_test_live.py --show와 같은 그림)와 파지 후보 그립
 * 모양까지 얹혀 있다(web/ros_bridge.py 참조). 뎁스 뷰만 원본 그대로다.
 *
 * **온디맨드 전환 이후**(docs/on-demand-perception.md) perception은 명령을 받은
 * 순간에만 관측하므로, 이 화면은 명령 사이에는 멈춰 있는 것이 정상이다 — 예전의
 * 주기 발행(0.5초)과 달리 "화면이 안 움직인다"가 고장 신호가 아니다. 그래서
 * 마지막 관측이 언제였는지를 같이 보여준다(worldStamp) — 안 그러면 사용자가 멈춘
 * 그림을 오래된 것으로 오해하거나, 반대로 방금 온 것을 못 알아본다.
 */
function Stream({ src, label }: { src: string; label: string }) {
  const [failed, setFailed] = useState(false)

  if (failed) {
    return (
      <div className="view">
        {label}
        <br />
        (스트림 연결 안 됨 — mock 모드이거나 카메라 미연결)
      </div>
    )
  }
  return <img className="view" src={src} alt={label} onError={() => setFailed(true)} />
}

function useAgoLabel(stamp: { sec: number; nanosec: number } | null): string {
  const [, setTick] = useState(0)

  useEffect(() => {
    if (!stamp) return
    const id = window.setInterval(() => setTick((n) => n + 1), 1000)
    return () => window.clearInterval(id)
  }, [stamp])

  if (!stamp) return '관측 없음'
  const observedAtMs = stamp.sec * 1000 + stamp.nanosec / 1e6
  const ageS = Math.max(0, (Date.now() - observedAtMs) / 1000)
  if (ageS < 2) return '방금 관측'
  if (ageS < 60) return `${Math.floor(ageS)}초 전 관측`
  return `${Math.floor(ageS / 60)}분 전 관측`
}

export default function CameraViews({
  worldStamp,
}: {
  worldStamp: { sec: number; nanosec: number } | null
}) {
  const agoLabel = useAgoLabel(worldStamp)

  return (
    <div>
      <div className="views">
        <Stream src="/api/camera/color" label="관측 뷰 (마스크 + 파지 후보)" />
        <Stream src="/api/camera/depth" label="뎁스 뷰" />
      </div>
      <div className="empty" style={{ padding: '6px 0', textAlign: 'left' }}>{agoLabel}</div>
    </div>
  )
}
