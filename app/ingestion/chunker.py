from __future__ import annotations

import hashlib
import re
import uuid
from typing import Protocol

from app.models import Chunk, Document, Section


class Chunker(Protocol):
    @property
    def component_id(self) -> str: ...

    def split(self, document: Document) -> list[Chunk]: ...


class HeadingChunker:
    def __init__(self, target_chars: int = 500, overlap_chars: int = 80) -> None:
        if target_chars < 100:
            raise ValueError("target_chars must be at least 100")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be in [0, target_chars)")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    @property
    def component_id(self) -> str:
        return f"heading-{self.target_chars}-{self.overlap_chars}-v1"

    @staticmethod
    def _split_long_block(text: str, limit: int) -> list[str]:
        if len(text) <= limit:
            return [text]
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；])", text) if item.strip()]
        if len(sentences) <= 1:
            return [text[index : index + limit] for index in range(0, len(text), limit)]
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > limit:
                parts.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            parts.append(current)
        return parts

    def _pack_section(self, section: Section) -> list[str]:
        expanded: list[str] = []
        for block in section.blocks:
            expanded.extend(self._split_long_block(block, self.target_chars))

        packed: list[str] = []
        current: list[str] = []
        current_size = 0
        for block in expanded:
            addition = len(block) + (1 if current else 0)
            if current and current_size + addition > self.target_chars:
                packed.append("\n".join(current))
                overlap = packed[-1][-self.overlap_chars :] if self.overlap_chars else ""
                current = [overlap, block] if overlap else [block]
                current_size = sum(len(item) for item in current) + len(current) - 1
            else:
                current.append(block)
                current_size += addition
        if current:
            packed.append("\n".join(current))
        return packed

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            for part_index, body in enumerate(self._pack_section(section), start=1):
                path_text = " > ".join([*document.breadcrumb, *section.path])
                text_lines = [f"车型：{document.vehicle_model}"]
                if document.manual_name:
                    text_lines.append(f"手册版本：{document.manual_name}")
                text_lines.extend(
                    [
                        f"路径：{path_text}",
                        f"标题：{section.title}",
                        f"正文：{body}",
                    ]
                )
                text = "\n".join(text_lines)
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                stable_key = "|".join(
                    [
                        document.manual_id,
                        document.snapshot_id,
                        document.topic_id,
                        "/".join(section.path),
                        str(part_index),
                        text_hash,
                    ]
                )
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        manual_id=document.manual_id,
                        snapshot_id=document.snapshot_id,
                        vehicle_model=document.vehicle_model,
                        manual_key=document.manual_key,
                        manual_name=document.manual_name,
                        manual_version=document.manual_version,
                        topic_id=document.topic_id,
                        title=document.title,
                        text=text,
                        section_path=section.path,
                        source_url=document.source_url,
                        content_hash=text_hash,
                        metadata={
                            "chunker": self.component_id,
                            "part_index": part_index,
                            "images": [image.model_dump() for image in section.images],
                        },
                    )
                )
        return chunks
