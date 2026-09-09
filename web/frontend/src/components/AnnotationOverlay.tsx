/** 데이터셋 상세 모달의 어노테이션 오버레이 (DatasetPage.tsx).
 *
 * points는 YOLO 세그멘테이션 좌표 그대로 — 이미지 크기 기준 정규화(0~1)다. viewBox를
 * "0 0 1 1"로 두고 부모 컨테이너(이미지와 같은 박스, absolute inset:0)에 꽉 채우면
 * 이미지 렌더 크기와 무관하게 좌표 변환 없이 그대로 맞는다 — 부모가 이미지의 실제
 * 렌더 박스(원본 비율 유지)와 정확히 같은 크기여야 한다(DatasetPage의 감싸는 div 참조).
 */
import type { AnnotationPolygon } from '../types'

const PALETTE = ['#e06c75', '#61afef', '#98c379', '#e5c07b', '#c678dd', '#56b6c2', '#d19a66']

export default function AnnotationOverlay({ polygons }: { polygons: AnnotationPolygon[] }) {
  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {polygons.map((poly, i) => {
        const color = PALETTE[i % PALETTE.length]
        const pointsAttr = poly.points.map(([x, y]) => `${x},${y}`).join(' ')
        const [labelX, labelY] = poly.points[0] ?? [0, 0]
        return (
          <g key={i}>
            <polygon
              points={pointsAttr}
              fill={color} fillOpacity={0.2}
              stroke={color} strokeWidth={0.0025}
            />
            <text
              x={labelX} y={Math.max(labelY - 0.008, 0.025)}
              fontSize={0.028} fill={color}
              stroke="#000" strokeWidth={0.006} paintOrder="stroke"
            >
              {poly.name_ko || poly.class_name || '?'}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
