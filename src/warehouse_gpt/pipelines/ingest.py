"""Download the raw Olist CSVs into ``data/raw`` (no Kaggle credentials needed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import structlog

from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.pipelines.tables import SOURCES

log = structlog.get_logger(__name__)


def download_raw(settings: Settings | None = None, force: bool = False) -> Path:
    settings = settings or get_settings()
    raw_dir = settings.raw_dir
    expected = {t.raw_file for t in SOURCES.values()}
    if not force and raw_dir.exists() and expected <= {p.name for p in raw_dir.iterdir()}:
        log.info("raw.cached", path=str(raw_dir))
        return raw_dir

    import kagglehub

    src = Path(kagglehub.dataset_download(settings.kaggle_dataset))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        shutil.copy2(src / name, raw_dir / name)
    log.info("raw.downloaded", path=str(raw_dir), files=len(expected))
    return raw_dir
