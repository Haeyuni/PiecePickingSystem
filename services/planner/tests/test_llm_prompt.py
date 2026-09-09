"""LLM 프롬프트 도메인 분기 테스트 (stdlib unittest — test_label_marks.py와 같은 이유).

VLM을 부르지 않고 `build_user_prompt`의 출력만 본다 — 도메인(가정/약국/재활용)별로
계획 해석 규칙이 [장면 맥락]에 들어가는지 확인한다.
"""
import unittest

from src import llm_client

WORLD = "검출된 물체: [이부프로펜정, 비타민C]"


class LlmPromptTest(unittest.TestCase):
    def test_general_has_no_domain_context(self):
        prompt = llm_client.build_user_prompt("약 줘", WORLD)
        self.assertNotIn("장면 맥락(도메인)", prompt)

    def test_pharmacy_infers_function_from_drug_name(self):
        """약국: 증상으로 온 지시를 약 이름에서 효능을 추론해 해결한다는 규칙이 들어간다."""
        prompt = llm_client.build_user_prompt("머리가 아플 때 먹는 약 줘", WORLD,
                                              domain="pharmacy")
        self.assertIn("[장면 맥락(도메인)]", prompt)
        self.assertIn("약 이름", prompt)
        self.assertIn("효능", prompt)

    def test_pharmacy_excludes_non_oral_products(self):
        """2026-09-09 실물: '머리가 아플 때 먹는 약'에 밴드(medica_band)를 잘못 승인한
        사고 대응 — 먹는 약이 아닌 외용/처치용품(밴드·연고·안약)을 먼저 거르는 규칙."""
        prompt = llm_client.build_user_prompt("머리가 아플 때 먹는 약 줘", WORLD,
                                              domain="pharmacy")
        self.assertIn("먹는 약", prompt)
        self.assertIn("medica_band", prompt)
        self.assertIn("검산", prompt)

    def test_recycle_classifies_by_material(self):
        """재활용: 재질별 분리를 목적으로 한다는 규칙이 들어간다."""
        prompt = llm_client.build_user_prompt("플라스틱 왼쪽, 캔 오른쪽", WORLD,
                                              domain="recycle")
        self.assertIn("[장면 맥락(도메인)]", prompt)
        self.assertIn("재질별 분리", prompt)


if __name__ == "__main__":
    unittest.main()