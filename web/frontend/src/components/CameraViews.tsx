import { useState } from 'react'

/** 카메라·뎁스 뷰 (화면정의서.md 2.2.4절)
 *
 * MJPEG(개발계획 D-5 1차 범위)를 <img>에 그대로 물린다 — 브라우저가
 * multipart/x-mixed-replace를 네이티브로 처리하므로 별도 스트리밍 클라이언트가
 * 필요 없다. mock 모드나 카메라 미연결 시 백엔드가 204를 돌려주는데, <img>는 204를
 * 에러로 보고 onError를 발화한다 — 그때만 안내 문구로 대체한다(깨진 아이콘 대신).
 *
 * 세그멘테이션 마스크는 오버레이하지 않는다 — 마스크는 grasp 내부 데이터다. 원본
 * 스트림만 보여준다.
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

export default function CameraViews() {
  return (
    <div className="views">
      <Stream src="/api/camera/color" label="카메라 뷰" />
      <Stream src="/api/camera/depth" label="뎁스 뷰" />
    </div>
  )
}
