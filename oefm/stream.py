"""Download, tile (v3), delete: the global pipeline stores only tiles (decision log "Tiling v3").

Batches of planned granules (oldest first) are downloaded and verified (oefm.download, unchanged),
tiled into catalog_v1 (oefm.tiling3), and the raw file is deleted once its tiling is recorded as
done. The manifest keeps the row with status 'tiled' (sha256, size, URL), so any granule can be
fetched again later. A granule whose tiling fails keeps its raw file (status stays 'verified') and is
retried on the next run. Resumable: rerun after a crash or the 48 h limit.

    python -m oefm stream [--products ...] [--batch 200] [--connections 8] [--workers 12] [--keep-raw]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _args(batch: int, connections: int, start=None, end=None) -> argparse.Namespace:
    return argparse.Namespace(connections=connections, dry_run=False, end=end, max_files=batch, per_product=0,
                              recheck=False, region=None, retries=5, retry_sleep=30.0, start=start, timeout=300.0)


def delete_tiled(cfg, rows: list[dict], log=print) -> tuple[int, int]:
    """Delete the raw files of rows whose v3 tiling is done; manifest status -> 'tiled'."""
    from .manifest import Manifest
    from .tiling3 import done_keys, v3_db

    db = v3_db(cfg)
    done = done_keys(db)
    db.close()
    man = Manifest(cfg.manifest_path)
    n, freed = 0, 0
    for r in rows:
        if r["granule_key"] not in done:
            continue
        p = cfg.data_root / r["local_path"]
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
        man.update(r["granule_key"], status="tiled", bytes_on_disk=0)
        n += 1
    man.close()
    log(f"stream: deleted {n:,} raw files ({freed / 1e9:,.1f} GB) after tiling")
    return n, freed


def run(cfg, products=None, batch: int = 200, connections: int = 8, workers: int = 0, keep_raw: bool = False,
        start=None, end=None, max_batches: int = 0, log=print) -> None:
    from . import download, tiling3
    from .manifest import Manifest

    prods = cfg.select_products(products) if products else cfg.satellite_products
    prods = {k: v for k, v in prods.items() if v.get("kind") == "satellite"}
    nb = 0
    while True:
        # 1. tile (and delete) anything verified but not yet tiled: leftovers of a previous run
        man = Manifest(cfg.manifest_path)
        verified = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(prods))})",
                                              tuple(prods))]
        man.close()
        if verified:
            tiling3.run(cfg, rows=verified, workers=workers, log=log)
            if not keep_raw:
                delete_tiled(cfg, verified, log)
        # 2. next batch
        if max_batches and nb >= max_batches:
            break
        stats = download.run(cfg, prods, _args(batch, connections, start, end), log)
        nb += 1
        if stats.get("files") == 0:
            log("stream: nothing left to download")
            break
        if not stats.get("ok"):
            log(f"stream: no file downloaded in this batch ({stats.get('failed', 0)} failed); stopping, check the log")
            break
    tiling3.export_index(cfg, log)
