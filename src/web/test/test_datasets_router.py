"""GET /api/datasets, GET /api/datasets/{item_id}/image — 데이터셋 브라우징 화면 API.

planner가 쓴 dataset_items를 읽기만 하는 라우터라, 실제 DB 대신 store 함수를 가짜로
바꾼다(routers/stt.py 테스트와 같은 패턴 — 전체 앱의 lifespan 없이 라우터만 올린다).
"""
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web import store
from web.routers import datasets as datasets_router

app = FastAPI()
app.include_router(datasets_router.router)
client = TestClient(app)


def item(item_id="item-1", class_name="toothpaste"):
    return {
        "item_id": item_id, "trace_id": "tr-1", "captured_at": "2026-09-08T12:00:00+00:00",
        "image_path": "/data/datasets/2026-09-08/tr-1.png",
        "label_path": "/data/datasets/2026-09-08/tr-1.json",
        "class_name": class_name, "name_ko": "치약", "attr_source": "llm_suggested",
        "confidence": 0.9, "reasoning": "흔한 치약 튜브 형태", "reviewed": False,
    }


class DatasetsRouterTest(unittest.TestCase):
    def setUp(self):
        self._real_query = store.query_dataset_items
        self._real_image = store.get_dataset_item_image_path

    def tearDown(self):
        store.query_dataset_items = self._real_query
        store.get_dataset_item_image_path = self._real_image

    def test_list_passes_filters_through_to_store(self):
        calls = []
        store.query_dataset_items = lambda **kw: calls.append(kw) or [item()]

        response = client.get("/api/datasets?class_name=toothpaste&limit=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [item()])
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
            store.get_dataset_item_image_path = lambda item_id: str(image_path)

            response = client.get("/api/datasets/item-1/image")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, content)

    def test_image_404_when_item_id_unknown(self):
        store.get_dataset_item_image_path = lambda item_id: None

        response = client.get("/api/datasets/no-such-item/image")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
