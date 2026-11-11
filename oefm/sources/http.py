"""Static files fetched over plain HTTPS (GEBCO)."""

from __future__ import annotations

from pathlib import PurePosixPath

import requests

from ..config import Config


def inventory(product: dict, cfg: Config, log=print) -> list[dict]:
    url = product["url"]
    filename = PurePosixPath(url.split("?")[0]).name
    size = None
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        size = int(r.headers.get("Content-Length") or 0) or None
    except Exception as exc:  # noqa: BLE001
        log(f"  {product['key']}: HEAD {url} failed ({exc}); size unknown")
    dom = cfg.domain()
    log(f"  {product['key']:18s} 1 file, {size or 0:,} bytes")
    return [{
        "granule_key": f"{product['key']}:{filename}",
        "product": product["key"],
        "kind": "physical",
        "family": product.get("variable", ""),
        "source": "http",
        "short_name": product.get("version"),
        "version": product.get("version"),
        "provider": PurePosixPath(url).parts[1] if "//" in url else None,
        "concept_id": None,
        "granule_ur": None,
        "filename": filename,
        "url": url,
        "time_start": None,
        "time_end": None,
        "west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0,
        "footprint": None,
        "size_bytes": size,
        "size_exact": 1 if size else 0,
        "checksum_algo": None,
        "checksum_expected": None,
        "request": None,
        "local_path": f"raw/{product['key']}/{filename}",
        "regions": ",".join(sorted(cfg.regions)),
        "windows": "static",
    }]
