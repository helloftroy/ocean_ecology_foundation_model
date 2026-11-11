"""Resumable, retry-safe downloads driven by the manifest.

* Each file is written to <name>.part and renamed only after it is complete and verified.
* An interrupted .part is resumed with an HTTP Range request where the server allows it.
* A file whose manifest status is 'verified' and that is still on disk at the right size is
  never fetched again; re-running a job therefore only does what is left.
* Every completed transfer is timed and stored in the manifest's `transfers` table; each run's
  aggregate throughput goes to `runs` (used for the benchmark and ETA in the report).
"""

from __future__ import annotations

import hashlib
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import Config
from .manifest import Manifest, utcnow
from .sources import cmems, nasa

CHUNK = 8 * 1024 * 1024
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAGIC = {
    b"\x89HDF\r\n\x1a\n": "hdf5/netcdf4",
    b"CDF\x01": "netcdf3",
    b"CDF\x02": "netcdf3-64bit",
    b"CDF\x05": "netcdf3-cdf5",
    b"PK\x03\x04": "zip",
}


class DownloadError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- verification
def sniff(path: Path) -> str | None:
    with open(path, "rb") as fh:
        head = fh.read(8)
    for magic, kind in MAGIC.items():
        if head.startswith(magic):
            return kind
    return None


def file_hashes(path: Path, algos=("sha256",)) -> dict[str, str]:
    hs = {a: hashlib.new(a) for a in algos}
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            for h in hs.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hs.items()}


def verify(row: dict, path: Path, hashes: dict[str, str] | None = None, expected_len: int | None = None) -> dict[str, Any]:
    """Check size, file signature and (where the archive publishes one) the checksum."""
    if not path.exists():
        raise DownloadError("file missing")
    size = path.stat().st_size
    served = expected_len or row.get("server_bytes")
    note = None
    if served and size != served:
        raise DownloadError(f"size {size} != server Content-Length {served}")
    if row["size_exact"] and row["size_bytes"] and size != row["size_bytes"]:
        if not served:
            raise DownloadError(f"size {size} != archive size {row['size_bytes']}")
        # The server's Content-Length is authoritative; CMR metadata can be stale (seen for
        # MODIS Aqua files listed under two GranuleURs). Record the difference, keep the file.
        note = f"archive metadata size {row['size_bytes']} differs from served size {size}"
    kind = sniff(path)
    if kind is None:
        with open(path, "rb") as fh:
            head = fh.read(300)
        hint = " (looks like a web page: login or app approval problem?)" if head.lstrip()[:1] == b"<" else ""
        raise DownloadError(f"not a NetCDF/HDF5/zip file{hint}")
    algo = (row["checksum_algo"] or "").lower() or None
    algos = ["sha256"] + ([algo] if algo and algo in hashlib.algorithms_available and algo != "sha256" else [])
    if hashes is None or any(a not in hashes for a in algos):
        hashes = file_hashes(path, algos)
    verified = 0
    if algo and row["checksum_expected"]:
        if hashes.get(algo, "").lower() != str(row["checksum_expected"]).lower():
            raise DownloadError(f"{algo} mismatch: {hashes.get(algo)} != {row['checksum_expected']}")
        verified = 1
    return {"bytes_on_disk": size, "sha256": hashes["sha256"], "checksum_verified": verified, "format": kind,
            "server_bytes": served, "size_note": note}


# ----------------------------------------------------------------------------- HTTP transfer
def http_fetch(url: str, dest: Path, sess, timeout: float, algos: list[str]) -> tuple[int, dict[str, str] | None, int]:
    """Stream url to dest.part (resuming), rename to dest. Returns (bytes_this_session, hashes, total_len)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with sess.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as r:
        if r.status_code == 416 and have:  # .part already complete (or bigger than the file): start over
            part.unlink()
            return http_fetch(url, dest, sess, timeout, algos)
        if r.status_code in RETRY_STATUS:
            raise DownloadError(f"HTTP {r.status_code} (retryable)")
        if r.status_code >= 400:
            raise DownloadError(f"HTTP {r.status_code}: {r.text[:200]}")
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise DownloadError("server returned a web page instead of the file (Earthdata login / app approval?)")
        resumed = r.status_code == 206 and have > 0
        if r.status_code == 206:
            total = int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0)
        else:
            total = int(r.headers.get("Content-Length") or 0)
            have = 0
        hs = None if resumed else {a: hashlib.new(a) for a in algos}
        got = 0
        with open(part, "ab" if resumed else "wb") as out:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                out.write(chunk)
                got += len(chunk)
                if hs is not None:
                    for h in hs.values():
                        h.update(chunk)
    if total and part.stat().st_size != total:
        raise DownloadError(f"incomplete: {part.stat().st_size} of {total} bytes (will resume)")
    part.replace(dest)
    return got, ({a: h.hexdigest() for a, h in hs.items()} if hs else None), total


# ----------------------------------------------------------------------------- driver
def select(man: Manifest, products: list[str], args) -> list[dict]:
    where = [f"product IN ({','.join('?' * len(products))})"]
    params: list[Any] = list(products)
    if not args.recheck and not args.per_product:
        where.append("status != 'verified'")
    if args.start:
        where.append("(time_end IS NULL OR substr(time_end,1,10) >= ?)")
        params.append(args.start)
    if args.end:
        where.append("(time_start IS NULL OR substr(time_start,1,10) <= ?)")
        params.append(args.end)
    if args.region:
        where.append("(',' || regions || ',') LIKE ?")
        params.append(f"%,{args.region},%")
    rows = [dict(r) for r in man.rows(" AND ".join(where), tuple(params))]
    if args.per_product:
        # Smoke test: the first N files of each product, preferring a date with all sensors.
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["product"], [])
            if len(out[r["product"]]) < args.per_product:
                out[r["product"]].append(r)
        rows = [r for v in out.values() for r in v
                if args.recheck or r["status"] != "verified"]  # the same N files on every re-run
    if args.max_files:
        rows = rows[: args.max_files]
    return rows


def run(cfg: Config, products: dict[str, dict], args, log=print) -> dict[str, Any]:
    man = Manifest(cfg.manifest_path)
    todo = select(man, list(products), args)

    # Files already on disk and verified elsewhere (e.g. a previous run that died before the
    # manifest update) are picked up by the verification pass below without a download.
    total_bytes = sum(r["size_bytes"] or 0 for r in todo)
    log(f"{len(todo):,} files to fetch or verify, about {total_bytes / 1e9:,.1f} GB, with {args.connections} connections")
    if args.dry_run or not todo:
        man.close()
        return {"files": len(todo), "bytes": total_bytes}

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    host = socket.gethostname()
    started = time.time()
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "bytes": 0}

    def one(row: dict) -> None:
        key = row["granule_key"]
        dest = cfg.data_root / row["local_path"]
        t0 = time.time()
        got = 0
        error = None
        info: dict[str, Any] = {}
        for attempt in range(1, args.retries + 1):
            try:
                expected_len = None
                hashes = None
                if dest.exists() and (not row["size_exact"] or dest.stat().st_size == row["size_bytes"]):
                    pass  # on disk already: verify only
                elif row["source"] == "copernicus_marine":
                    man.update(key, status="downloading", attempts=(row["attempts"] or 0) + attempt)
                    cmems.download(row, dest)
                    got = dest.stat().st_size
                else:
                    man.update(key, status="downloading", attempts=(row["attempts"] or 0) + attempt)
                    sess = nasa.session() if row["source"] == "nasa_earthdata" else _plain_session()
                    algo = (row["checksum_algo"] or "").lower()
                    algos = ["sha256"] + ([algo] if algo in hashlib.algorithms_available and algo not in ("", "sha256") else [])
                    n, hashes, expected_len = http_fetch(row["url"], dest, sess, args.timeout, algos)
                    got += n
                info = verify(row, dest, hashes, expected_len)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - every failure is retried, then recorded
                error = f"{type(exc).__name__}: {exc}"[:500]
                if isinstance(exc, DownloadError) and dest.exists():
                    dest.unlink()  # a complete file that failed verification: fetch it cleanly next attempt
                if attempt < args.retries:
                    time.sleep(min(args.retry_sleep * 2 ** (attempt - 1), 600))
        secs = time.time() - t0
        if error is None:
            man.update(key, status="verified", last_error=None, download_seconds=secs, downloaded_at=utcnow(),
                       bytes_on_disk=info["bytes_on_disk"], sha256=info["sha256"],
                       checksum_verified=info["checksum_verified"], server_bytes=info["server_bytes"],
                       size_note=info["size_note"])
        else:
            man.update(key, status="failed", last_error=error, download_seconds=secs)
        if got:
            man.add_transfer(run_id=run_id, granule_key=key, product=row["product"], source=row["source"], host=host,
                             bytes=got, seconds=secs, started_at=utcnow(), ok=int(error is None))
        with lock:
            stats["ok" if error is None else "failed"] += 1
            stats["bytes"] += got
            done = stats["ok"] + stats["failed"]
            el = max(time.time() - started, 1e-6)
            rate = stats["bytes"] / el
            left = total_bytes - stats["bytes"]
            eta = left / rate / 3600 if rate > 0 else float("nan")
            log(f"[{done}/{len(todo)}] {'ok  ' if error is None else 'FAIL'} {row['product']} {row['filename']} "
                f"{got / 1e6:,.1f} MB in {secs:,.0f}s | run {stats['bytes'] / 1e9:,.2f} GB at {rate / 1e6:,.1f} MB/s, "
                f"~{eta:,.1f} h left" + (f" | {error}" if error else ""))

    # Copernicus Marine subsets are server-side jobs; keep them to a couple at a time.
    http_rows = [r for r in todo if r["source"] != "copernicus_marine"]
    cm_rows = [r for r in todo if r["source"] == "copernicus_marine"]
    with ThreadPoolExecutor(max_workers=max(1, args.connections)) as pool, \
            ThreadPoolExecutor(max_workers=max(1, min(2, args.connections))) as cm_pool:
        futures = [pool.submit(one, r) for r in http_rows] + [cm_pool.submit(one, r) for r in cm_rows]
        for f in as_completed(futures):
            f.result()

    wall = time.time() - started
    man.save_run(run_id=run_id, host=host, products=",".join(products), connections=args.connections,
                 started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), finished_at=utcnow(),
                 files_ok=stats["ok"], files_failed=stats["failed"], bytes=stats["bytes"], wall_seconds=wall)
    log(f"Run {run_id}: {stats['ok']} ok, {stats['failed']} failed, {stats['bytes'] / 1e9:,.2f} GB in {wall / 60:,.1f} min "
        f"({stats['bytes'] / max(wall, 1e-6) / 1e6:,.1f} MB/s aggregate). Failed files are retried on the next run.")
    man.close()
    return stats


_plain = threading.local()


def _plain_session():
    import requests

    s = getattr(_plain, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "oefm-acquisition/0.1"
        _plain.s = s
    return s
