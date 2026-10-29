from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_assistant.knowledge import KnowledgeBase, chunk_text


class KnowledgeBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = KnowledgeBase.from_directory(ROOT / "data")

    def test_loads_all_documents(self) -> None:
        self.assertEqual(4, len(self.knowledge.documents))
        self.assertGreater(self.knowledge.chunk_count, 4)

    def test_annual_leave_search(self) -> None:
        hits = self.knowledge.search("年假有多少天，怎么申请？", top_k=3)
        self.assertTrue(hits)
        self.assertIn("带薪年假", hits[0].chunk.section)
        self.assertIn("5 天", hits[0].chunk.text)

    def test_hotel_standard_search(self) -> None:
        hits = self.knowledge.search("上海出差住宿每晚最多报销多少钱", top_k=3)
        self.assertTrue(hits)
        self.assertEqual("travel-expense-policy.md", hits[0].chunk.source)
        self.assertIn("650", hits[0].chunk.text)

    def test_leak_response_search(self) -> None:
        hits = self.knowledge.search("数据泄露后要打什么电话", top_k=3)
        self.assertTrue(hits)
        self.assertEqual("information-security.md", hits[0].chunk.source)
        self.assertIn("400-800-2718", hits[0].chunk.text)

    def test_chunk_text_respects_size_and_overlap(self) -> None:
        text = "。" .join(["这是一个用于测试自然语言分块的句子"] * 80) + "。"
        chunks = chunk_text(text, max_chars=180, overlap=30)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 181 for chunk in chunks))
        self.assertTrue(any(chunks[0][-20:] in chunks[1] for _ in [0]))

    def test_empty_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeBase.from_directory(ROOT / "tests" / "empty-kb")


if __name__ == "__main__":
    unittest.main()

