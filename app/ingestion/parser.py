from __future__ import annotations

import hashlib
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from app.models import Document, ImageRef, Section, TopicRef


def _normalize(value: str | None) -> str:
    return " ".join(str(value or "").replace("\u2028", " ").split()).strip()


def _direct(node: Tag, name: str, **attrs: object) -> Tag | None:
    found = node.find(name, recursive=False, **attrs)
    return found if isinstance(found, Tag) else None


def _render_table(node: Tag) -> list[str]:
    rows: list[str] = []
    for row in node.find_all("tr"):
        cells = [_normalize(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    if rows:
        return ["\n".join(rows)]
    text = _normalize(node.get_text(" ", strip=True))
    return [text] if text else []


def _render_block(
    node: Tag,
    *,
    topic_url: str,
    section_path: list[str],
    images: list[ImageRef],
) -> list[str]:
    name = node.name
    if name == "p":
        text = _normalize(node.get_text(" ", strip=True))
        return [text] if text else []
    if name in {"ul", "ol"}:
        rendered: list[str] = []
        for index, item in enumerate(node.find_all("li", recursive=False), start=1):
            text = _normalize(item.get_text(" ", strip=True))
            if text:
                marker = f"{index}." if name == "ol" else "-"
                rendered.append(f"{marker} {text}")
        return rendered
    if name == "table":
        return _render_table(node)
    if name == "figure":
        image = node.find("img")
        if isinstance(image, Tag) and image.get("src"):
            images.append(
                ImageRef(
                    url=urljoin(topic_url, str(image.get("src"))),
                    alt=_normalize(image.get("alt")),
                    section_path=section_path,
                )
            )
        return []
    if name in {"div", "section"}:
        classes = set(node.get("class") or [])
        prefix = ""
        if "warning" in classes or "note_warning" in classes:
            prefix = "[警告] "
        elif "caution" in classes or "note_caution" in classes:
            prefix = "[注意] "
        children: list[str] = []
        for child in node.find_all(recursive=False):
            if isinstance(child, Tag):
                children.extend(
                    _render_block(
                        child,
                        topic_url=topic_url,
                        section_path=section_path,
                        images=images,
                    )
                )
        if prefix and children:
            children[0] = prefix + children[0]
        return children
    text = _normalize(node.get_text(" ", strip=True))
    return [text] if text else []


class LiXiangHtmlParser:
    def parse(
        self,
        html: bytes,
        topic: TopicRef,
        *,
        manual_id: str,
        snapshot_id: str,
        vehicle_model: str,
    ) -> Document | None:
        soup = BeautifulSoup(html, "lxml")
        article = soup.select_one("main article[role='article']") or soup.select_one(
            "article[role='article']"
        )
        if not isinstance(article, Tag):
            return None

        sections: list[Section] = []

        def walk(current: Tag, parent_path: list[str], fallback_level: int) -> None:
            heading = current.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
            heading_text = heading.get_text(" ", strip=True) if isinstance(heading, Tag) else ""
            title = _normalize(heading_text)
            level = int(heading.name[1]) if isinstance(heading, Tag) else fallback_level
            path = [*parent_path, title] if title else list(parent_path)
            body = _direct(current, "div", class_="body")
            blocks: list[str] = []
            images: list[ImageRef] = []
            if body is not None:
                for child in body.find_all(recursive=False):
                    if isinstance(child, Tag):
                        blocks.extend(
                            _render_block(
                                child,
                                topic_url=topic.source_url,
                                section_path=path,
                                images=images,
                            )
                        )
            if blocks or images:
                sections.append(
                    Section(
                        title=title or topic.title,
                        level=level,
                        path=path or [topic.title],
                        blocks=blocks,
                        images=images,
                    )
                )
            for nested in current.find_all("article", class_="topic", recursive=False):
                if isinstance(nested, Tag):
                    walk(nested, path, level + 1)

        walk(article, [], 1)
        if not sections:
            return None

        content_hash = hashlib.sha256(html).hexdigest()
        document_id = f"{manual_id}:{snapshot_id}:{topic.topic_id}"
        return Document(
            document_id=document_id,
            manual_id=manual_id,
            snapshot_id=snapshot_id,
            vehicle_model=vehicle_model,
            topic_id=topic.topic_id,
            title=topic.title,
            breadcrumb=topic.breadcrumbs,
            source_url=topic.source_url,
            sections=sections,
            content_hash=content_hash,
            metadata={"source_file": topic.source_file},
        )
