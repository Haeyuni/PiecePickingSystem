"""GET /api/datasets, GET /api/datasets/{trace_id}/image·annotations, POST .../review —
데이터셋 브라우징+큐레이션 화면 API.

읽기(list/image/annotations)는 store 함수만 부르는 라우터라 store를 가짜로 바꾼다
(routers/stt.py 테스트와 같은 패턴 — 전체 앱의 lifespan 없이 라우터만 올린다). review는
planner로 넘기는 프록시라 planner_client를 가짜로 바꾼다.
"""
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web import planner_client, store
from web.routers import datasets as datasets_router

app = FastAPI()
app.include_router(datasets_router.router)
client = TestClient(app)


def image(trace_id="tr-1", class_name="toothpaste", reviewed=False):
    return {
        "trace_id": trace_id, "captured_at": "2026-09-08T12:00:00+00:00",
        "reviewed": reviewed,
        "objects": [{"class_name": class_name, "name_ko": "치약"}],
    }


class DatasetsRouterTest(unittest.TestCase):
    def setUp(self):
        self._real_query = store.query_dataset_items
        self._real_image = store.get_dataset_image_path
        self._real_polygons = store.get_dataset_item_polygons
        self._real_review = planner_client.review_dataset_item

    def tearDown(self):
        store.query_dataset_items = self._real_query
        store.get_dataset_image_path = self._real_image
        store.get_dataset_item_polygons = self._real_polygons
        planner_client.review_dataset_item = self._real_review

    def test_list_passes_filters_through_to_store(self):
        calls = []
        store.query_dataset_items = lambda **kw: calls.append(kw) or [image()]

        response = client.get("/api/datasets?class_name=toothpaste&limit=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [image()])
        self.assertEqual(calls[0]["class_name"], "toothpaste")
        self.assertEqual(calls[0]["limit"], 10)

    def test_list_defaults_to_no_filters(self):
        calls = []
        store.query_dataset_items = lambda **kw: calls.append(kw) or []

        client.get("/api/datasets")

        self.assertIsNone(calls[0]["class_name"])
        self.assertIsNone(calls[0]["since"])
        self.assertEqual(calls[0]["limit"], 50)

    def test_image_serves_the_file_when_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "tr-1.png"
            content = b"\x89PNG\r\n\x1a\n" + b"0" * 8
            image_path.write_bytes(content)
            store.get_dataset_image_path = lambda trace_id: str(image_path)

            response = client.get("/api/datasets/tr-1/image")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, content)

    def test_image_404_when_trace_id_unknown(self):
        store.get_dataset_image_path = lambda trace_id: None

        response = client.get("/api/datasets/no-such-trace/image")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "NOT_FOUND")

    def test_annotations_returns_polygons(self):
        polygons = [{"class_name": "toothpaste", "name_ko": "치약",
                    "points": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}]
        store.get_dataset_item_polygons = lambda trace_id: polygons

        response = client.get("/api/datasets/tr-1/annotations")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["polygons"], polygons)

    def test_annotations_404_when_none(self):
        store.get_dataset_item_polygons = lambda trace_id: None

        response = client.get("/api/datasets/tr-1/annotations")

        self.assertEqual(response.status_code, 404)

    def test_review_forwards_to_planner_client(self):
        calls = []

        async def fake_review(trace_id, approved):
            calls.append((trace_id, approved))
            return {"status_code": 200, "body": {"trace_id": trace_id, "approved": approved}}

        planner_client.review_dataset_item = fake_review

        response = client.post("/api/datasets/tr-1/review", json={"approved": True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [("tr-1", True)])

    def test_bulk_review_forwards_trace_id_list_to_planner_client(self):
        calls = []
        real_bulk = planner_client.bulk_review_dataset_items

        async def fake_bulk(trace_ids, approved):
            calls.append((trace_ids, approved))
            return {"status_code": 200,
                   "body": {"trace_ids": trace_ids, "approved": approved, "rows_affected": 4}}

        planner_client.bulk_review_dataset_items = fake_bulk
        try:
            response = client.post("/api/datasets/bulk-review",
                                   json={"trace_ids": ["tr-1", "tr-2"], "approved": False})
        finally:
            planner_client.bulk_review_dataset_items = real_bulk

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls, [(["tr-1", "tr-2"], False)])
        self.assertEqual(response.json()["rows_affected"], 4)


if __name__ == "__main__":
    unittest.main()
