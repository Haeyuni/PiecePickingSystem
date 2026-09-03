/** 카메라·뎁스 뷰 (화면정의서.md 2.2.4)
 *
 * 세그멘테이션 마스크는 오버레이하지 않는다 — 마스크는 grasp 내부 데이터다.
 * 스트림 전달 방식(MJPEG 등)은 개발계획 D-5 미확정이라, 그때까지 자리만 잡아둔다.
 */
export default function CameraViews() {
  return (
    <div className="views">
      <div className="view">카메라 뷰<br />(스트림 미연결 — D-5 확정 후 연결)</div>
      <div className="view">뎁스 뷰<br />(스트림 미연결)</div>
    </div>
  )
}
