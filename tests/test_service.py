from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from knowledge_assistant.config import Settings
from knowledge_assistant.knowledge import KnowledgeBase
from knowledge_assistant.llm import ModelRuntime
from knowledge_assistant.service import AssistantService


class AssistantServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = KnowledgeBase.from_directory(ROOT / "data")
        cls.settings = Settings(docs_dir=ROOT / "data", top_k=3)

    def test_demo_mode_returns_sources(self) -> None:
        runtime = ModelRuntime(
            llm=None,
            mode="demo",
            model_name="本地检索演示",
        )
        service = AssistantService(self.knowledge, self.settings, runtime)
        result = service.answer("年假怎么申请？")
        self.assertEqual("demo", result.mode)
        self.assertTrue(result.sources)
        self.assertIn("带薪年假", result.sources[0].section)
        self.assertIn("[资料1]", result.answer)
        self.assertIn("5 天带薪年假", result.answer)
        self.assertIn("提前 2 个工作日", result.answer)

    def test_langchain_chain_with_fake_model(self) -> None:
        llm = FakeListChatModel(
            responses=["年假需要提前 2 个工作日通过 OA 申请。[资料1]"]
        )
        runtime = ModelRuntime(llm=llm, mode="openai", model_name="fake-model")
        service = AssistantService(self.knowledge, self.settings, runtime)
        result = service.answer(
            "年假怎么申请？",
            history=[
                {"role": "user", "content": "我是正式员工。"},
                {"role": "assistant", "content": "已了解。"},
            ],
        )
        self.assertEqual("openai", result.mode)
        self.assertIn("[资料1]", result.answer)
        self.assertGreaterEqual(len(result.sources), 1)

    def test_empty_question_is_rejected(self) -> None:
        runtime = ModelRuntime(llm=None, mode="demo", model_name="demo")
        service = AssistantService(self.knowledge, self.settings, runtime)
        with self.assertRaises(ValueError):
            service.answer("   ")


if __name__ == "__main__":
    unittest.main()

