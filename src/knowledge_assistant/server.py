"""零前端构建依赖的本地 HTTP 服务。"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .config import Settings
from .knowledge import KnowledgeBase
from .llm import ModelConfigurationError, create_model_runtime
from .service import AssistantService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"
MAX_REQUEST_BYTES = 100_000


def create_app(
    settings: Settings | None = None,
    knowledge: KnowledgeBase | None = None,
) -> AssistantService:
    resolved = settings or Settings.from_env(PROJECT_ROOT)
    kb = knowledge or KnowledgeBase.from_directory(resolved.docs_dir)
    runtime = create_model_runtime(resolved)
    return AssistantService(kb, resolved, runtime=runtime)


def make_handler(service: AssistantService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Zhiwen/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", **service.status})
                return
            if path == "/api/knowledge":
                self._json(HTTPStatus.OK, service.status)
                return
            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/api/chat":
                self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在。"})
                return

            try:
                payload = self._read_json()
                question = str(payload.get("question", "")).strip()
                history = payload.get("history", [])
                if not isinstance(history, list):
                    raise ValueError("history 必须是数组。")
                normalized_history = [
                    {
                        "role": str(item.get("role", "")),
                        "content": str(item.get("content", "")),
                    }
                    for item in history[-8:]
                    if isinstance(item, dict)
                ]
                response = service.answer(question, normalized_history)
                self._json(HTTPStatus.OK, response.to_dict())
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                print("回答生成失败：", file=sys.stderr)
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "回答生成失败，请检查模型服务或查看终端日志。"},
                )

        def _read_json(self) -> dict[str, object]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("请求体不能为空。")
            if length > MAX_REQUEST_BYTES:
                raise ValueError("请求体过大。")
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise ValueError("请求类型必须是 application/json。")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("JSON 格式不正确。") from exc
            if not isinstance(data, dict):
                raise ValueError("请求体必须是 JSON 对象。")
            return data

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path == "/" else path.lstrip("/")
            candidate = (STATIC_DIR / relative).resolve()
            try:
                candidate.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "文件不存在。"})
                return
            if not candidate.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "文件不存在。"})
                return

            body = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {
                "application/javascript",
                "application/json",
            }:
                content_type += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            if not os.getenv("QUIET_HTTP"):
                super().log_message(format, *args)

    return Handler


def run() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings.from_env(PROJECT_ROOT)
    try:
        service = create_app(settings)
    except (ModelConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(service))
    print("知问企业知识库助手已启动")
    print(f"访问地址：http://{settings.host}:{settings.port}")
    print(
        f"模型模式：{service.runtime.mode} | "
        f"模型：{service.runtime.model_name} | "
        f"文档：{len(service.knowledge.documents)} | "
        f"分块：{service.knowledge.chunk_count}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()

