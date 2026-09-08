import { useEffect, useState } from 'react'

/** 카메라 뷰 (화면정의서.md 2.2.4절 개정, D-5, 2026-09-08 재개정)
 *
 * MJPEG(개발계획 D-5 1차 범위)를 <img>에 그대로 물린다 — 브라우저가
 * multipart/x-mixed-replace를 네이티브로 처리하므로 별도 스트리밍 클라이언트가
 * 필요 없다. mock 모드나 카메라 미연결 시 백엔드가 204를 돌려주는데, <img>는 204를
 * 에러로 보고 onError를 발화한다 — 그때만 안내 문구로 대체한다(깨진 아이콘 대신).
 *
 * 화면 기본(위, 크게)은 **관측 결과 뷰**(`/api/camera/color`, grasp가 합성한
 * /grasp/debug_image — 마스크 윤곽 + 파지 후보 그립 모양)다. 온디맨드 전환 이후
 * (docs/on-demand-perception.md) perception은 명령을 받은 순간에만 관측하므로, 이 뷰는
 * 명령 사이에는 마지막 결과가 그대로 남아 있는 사진처럼 멈춰 있는 게 정상이다 — 그래서
 * 마지막 관측이 언제였는지를 같이 보여준다(worldStamp). 그 아래(작게, 나란히) **원본
 * 카메라 뷰**(`/api/camera/raw`, 리얼센스 원본)와 **뎁스 뷰**(`/api/camera/depth`,
 * 컬러맵 시각화)를 참고용으로 둔다 — 둘 다 명령 사이에도 항상 실시간이라 카메라가
 * 살아있는지는 이쪽으로 바로 확인한다.
 *
 * 관측 결과 뷰는 `worldStamp`로 키를 준다 — 새 관측이 오기 전에 <img>가 한 번
 * onError(204/연결 끊김)로 "스트림 연결 안 됨" 문구로 넘어가면, 그 뒤 grasp가 실제로
 * 프레임을 발행해도 이미 <img> 대신 문구가 떠 있어 재시도하지 않는다(onError는 한
 * 번만 본다). key가 바뀌면 React가 그 자리의 컴포넌트를 통째로 새로 마운트하므로
 * `failed` 상태가 초기화되고 <img>가 새로 연결을 시도한다 — 브라우저 새로고침 없이도
 * 관측이 끝날 때마다 자동으로 다시 붙는다.
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
  const resultKey = worldStamp ? `${worldStamp.sec}-${worldStamp.nanosec}` : 'none'

  return (
    <div>
      <div className="views">
        <Stream key={resultKey} src="/api/camera/color" label="관측 결과 (마스크 + 파지 후보)" />
      </div>
      <div className="views views-mini">
        <Stream src="/api/camera/raw" label="카메라 뷰" />
        <Stream src="/api/camera/depth" label="뎁스 뷰" />
      </div>
      <div className="empty" style={{ padding: '6px 0', textAlign: 'left' }}>{agoLabel}</div>
    </div>
  )
}
