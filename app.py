"""知问企业知识库助手启动入口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from knowledge_assistant.server import run  # noqa: E402


if __name__ == "__main__":
    run()

