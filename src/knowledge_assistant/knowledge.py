"""知识库加载、分块和检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
SENTENCE_RE = re.compile(r"(?<=[。！？；.!?;])\s*|\n+")
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
QUERY_STOP_CHARS = set(
    "的了吗呢是在有和与及或把被对为请问如何多少几什么哪谁公司员工可以应该需要能否是否提供相关制度规定"
)
QUERY_STOP_TOKENS = {
    "公司",
    "员工",
    "可以",
    "应该",
    "需要",
    "是否",
    "提供",
    "相关",
    "制度",
    "规定",
    "如何",
    "多少",
    "什么",
    "怎么",
    "请问",
}


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    title: str
    source: str
    section: str
    text: str
    position: int

    @property
    def label(self) -> str:
        if self.section and self.section != self.title:
            return f"{self.title} › {self.section}"
        return self.title


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    title: str
    source: str
    chunk_count: int


class KnowledgeBase:
    """一个零外部服务依赖的中文友好检索索引。"""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("知识库为空，请至少在 data 目录中放入一个文档。")
        self._chunks = chunks
        self._tokens = [_tokenize(chunk.text + " " + chunk.section) for chunk in chunks]
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )
        self._documents = self._build_document_index()

    @classmethod
    def from_directory(cls, directory: Path) -> "KnowledgeBase":
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"知识库目录不存在：{directory}")

        chunks: list[KnowledgeChunk] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            source = path.relative_to(directory).as_posix()
            chunks.extend(_load_markdown(path, source))

        return cls(chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def documents(self) -> list[DocumentInfo]:
        return list(self._documents)

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        query_tokens = _query_tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        strong_query_tokens = {
            token for token in query_counts if len(token) > 1 or token.isascii()
        }
        total = len(self._chunks)
        scores: list[tuple[float, KnowledgeChunk]] = []

        for chunk, tokens in zip(self._chunks, self._tokens, strict=True):
            token_counts = Counter(tokens)
            if strong_query_tokens and not any(
                token in token_counts for token in strong_query_tokens
            ):
                continue
            score = 0.0
            for token, query_tf in query_counts.items():
                tf = token_counts.get(token, 0)
                if tf == 0:
                    continue
                df = self._document_frequency[token]
                idf = math.log((total + 1) / (df + 0.5)) + 1.0
                score += (1.0 + math.log(tf)) * idf * (1.0 + math.log(query_tf))

            title_tokens = set(_tokenize(chunk.section))
            overlap = len(title_tokens.intersection(query_counts))
            score += overlap * 0.8
            score /= math.sqrt(max(len(tokens), 1))
            scores.append((score, chunk))

        scores.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            SearchHit(chunk=chunk, score=score)
            for score, chunk in scores[: max(1, top_k)]
            if score > 0
        ]

    def _build_document_index(self) -> list[DocumentInfo]:
        grouped: dict[str, dict[str, object]] = {}
        for chunk in self._chunks:
            item = grouped.setdefault(
                chunk.source,
                {"title": chunk.title, "source": chunk.source, "chunks": 0},
            )
            item["chunks"] = int(item["chunks"]) + 1
        return [
            DocumentInfo(
                title=str(item["title"]),
                source=str(item["source"]),
                chunk_count=int(item["chunks"]),
            )
            for item in grouped.values()
        ]


def _load_markdown(path: Path, source: str) -> list[KnowledgeChunk]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    first_heading = HEADING_RE.search(raw)
    title = first_heading.group(1) if first_heading else path.stem
    sections = _split_sections(raw, title)
    chunks: list[KnowledgeChunk] = []
    position = 0

    for section_name, body in sections:
        for part in chunk_text(body, max_chars=520, overlap=80):
            position += 1
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{source}#{position}",
                    title=title,
                    source=source,
                    section=section_name,
                    text=part,
                    position=position,
                )
            )
    return chunks


def _split_sections(raw: str, default_title: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(raw))
    if not matches:
        return [(default_title, _clean_text(raw))]

    sections: list[tuple[str, str]] = []
    preface = raw[: matches[0].start()].strip()
    if preface:
        sections.append((default_title, _clean_text(preface)))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        name = match.group(1).strip()
        body = _clean_text(raw[match.end() : end])
        if body:
            sections.append((name, body))
    return sections


def chunk_text(text: str, max_chars: int = 520, overlap: int = 80) -> list[str]:
    """按自然边界切块，并保留少量重叠以减少上下文断裂。"""

    cleaned = _clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        hard_end = min(start + max_chars, len(cleaned))
        if hard_end == len(cleaned):
            end = hard_end
        else:
            soft_start = start + int(max_chars * 0.62)
            boundary_candidates = [
                cleaned.rfind(mark, soft_start, hard_end)
                for mark in ("。", "！", "？", "\n", "；", ";", "，", ",")
            ]
            boundary = max(boundary_candidates)
            end = boundary + 1 if boundary >= soft_start else hard_end

        part = cleaned[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """中文同时使用单字和双字，英文与数字按词切分。"""

    tokens: list[str] = []
    for segment in TOKEN_RE.findall(text.lower()):
        if "\u3400" <= segment[0] <= "\u9fff":
            tokens.extend(segment)
            tokens.extend(
                segment[index : index + 2] for index in range(len(segment) - 1)
            )
        else:
            tokens.append(segment)
    return tokens


def _query_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _tokenize(text):
        if token in QUERY_STOP_TOKENS:
            continue
        if len(token) == 1 and token in QUERY_STOP_CHARS:
            continue
        if (
            len(token) == 2
            and all("\u3400" <= char <= "\u9fff" for char in token)
            and any(char in QUERY_STOP_CHARS for char in token)
        ):
            continue
        tokens.append(token)
    return tokens


def excerpt_for_query(text: str, query: str, limit: int = 360) -> str:
    """从命中段落中按原文顺序选取相关信息。"""

    sentences = [item.strip() for item in SENTENCE_RE.split(text) if item.strip()]
    query_tokens = set(_query_tokenize(query))
    scored = [
        (len(query_tokens.intersection(_tokenize(sentence))), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    ranked = sorted(scored, reverse=True)
    relevant = [item for item in ranked[:3] if item[0] > 0]
    if scored and scored[0][0] > 0 and all(item[1] != 0 for item in relevant):
        relevant.append(scored[0])
    if relevant:
        selected = " ".join(item[2] for item in sorted(relevant, key=lambda item: item[1]))
    else:
        selected = sentences[0] if sentences else text
    if len(selected) > limit:
        selected = selected[: limit - 1].rstrip() + "…"
    return selected

