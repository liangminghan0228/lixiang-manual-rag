from __future__ import annotations

import json

import pytest

from app.ingestion.batch_service import container_for_manual, select_entries
from app.ingestion.catalog import parse_catalog
from app.wiring import build_container


def catalog_payload() -> bytes:
    return json.dumps(
        [
            {
                "id": 38,
                "url": (
                    "https://manuals.lixiang.com/zh-cn/M012020COMMON/20250813183937/index.html"
                ),
                "publishDate": "2025-08-13T18:39:37",
                "version": "3.4.8",
                "name": "理想ONE 2020款",
            },
            {
                "id": 494,
                "url": ("https://manuals.lixiang.com/zh-cn/W022025ULTRA/20260422091942/index.html"),
                "publishDate": "2026-04-22T09:19:42",
                "version": "8.5.0",
                "name": "理想i8",
            },
        ],
        ensure_ascii=False,
    ).encode()


def test_parse_and_select_official_catalog() -> None:
    entries = parse_catalog(b"\xef\xbb\xbf" + catalog_payload())

    assert len(entries) == 2
    assert entries[1].manual_key == "W022025ULTRA"
    assert entries[1].snapshot_id == "20260422091942"
    assert select_entries(entries, "i8|W02", None) == [entries[1]]


def test_catalog_rejects_untrusted_manual_url() -> None:
    payload = catalog_payload().replace(b"manuals.lixiang.com", b"example.invalid")
    with pytest.raises(ValueError, match="unsupported manual URL"):
        parse_catalog(payload)


def test_manual_container_reuses_models_and_changes_data_scope() -> None:
    base = build_container("configs/test.yaml")
    entry = parse_catalog(catalog_payload())[1]

    selected = container_for_manual(base, entry)

    assert selected.embedder is base.embedder
    assert selected.vector_store is base.vector_store
    assert selected.settings.data.manual_key == "W022025ULTRA"
    assert selected.crawler.output_dir.parts[-2:] == ("W022025ULTRA", "20260422091942")


def test_all_models_config_ignores_single_model_collection_override() -> None:
    container = build_container("configs/all-models.yaml", apply_runtime_overrides=False)

    assert container.settings.vector_store.collection == "lixiang_all_manuals_bge_m3_v1"
