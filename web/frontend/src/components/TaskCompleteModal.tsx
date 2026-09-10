/** 작업 완료 팝업 (계획의 모든 스텝이 끝났을 때 1회).
 *
 * 승인 모달과 달리 서버에 보낼 응답이 없다 — 계획이 끝났다는 사실만 알린다.
 * 스텝 중 실패가 섞여 있으면 몇 건이 실패했는지 같이 보여준다(진행 바만으로는
 * 화면을 계속 보고 있지 않으면 놓친다).
 */
import type { Trace } from '../types'

export default function TaskCompleteModal({
  trace, onClose,
}: {
  trace: Trace
  onClose: () => void
}) {
  const total = trace.steps.length
  const succeeded = trace.steps.filter((s) => s.status === 'success').length
  const failed = trace.steps.filter((s) => s.status === 'failure').length
  // 실패로 시퀀스가 중단되면 뒤 스텝은 pending으로 남는다(ControlPage의 완료 판정 참조).
  const skipped = total - succeeded - failed

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h3>{failed === 0 ? '작업 완료' : '작업 종료 (일부 실패)'}</h3>

        <div className="muted" style={{ fontSize: 12 }}>
          &ldquo;{trace.command_text}&rdquo;
        </div>

        <div className={`command-hint${failed === 0 ? ' hint-ok' : ' hint-error'}`}>
          {failed === 0
            ? `${total}개 스텝을 모두 완료했습니다.`
            : `${total}개 중 ${succeeded}개 성공, ${failed}개 실패` +
              (skipped > 0 ? `, ${skipped}개는 실행하지 못했습니다.` : '했습니다.')}
        </div>

        <div className="modal-actions">
          <button onClick={onClose}>확인</button>
        </div>
      </div>
    </div>
  )
}
