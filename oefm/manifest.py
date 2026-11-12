"""The acquisition manifest: one SQLite file recording every file we intend to have.

One row per distinct file (granule). A granule that intersects several regions or windows is
stored once; its `regions` and `windows` columns list all of them. Status moves:

    planned -> downloading -> downloaded -> verified
                         \\-> failed (retried on the next run)

Rows are only ever added or updated, never deleted, so an inventory re-run cannot lose the
download state of files already on disk. Run one download job per manifest at a time: SQLite
locking on GPFS is not reliable enough for several writers on different nodes.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

COLUMNS = {
    "granule_key": "TEXT PRIMARY KEY",   # <product>:<filename>
    "product": "TEXT NOT NULL",          # config key, e.g. pace_oci_aop
    "kind": "TEXT",                      # satellite | physical
    "family": "TEXT",
    "source": "TEXT",                    # nasa_earthdata | copernicus_marine | http
    "short_name": "TEXT",                # CMR short name or CMEMS dataset id
    "version": "TEXT",
    "provider": "TEXT",
    "concept_id": "TEXT",                # CMR granule concept id(s)
    "granule_ur": "TEXT",
    "filename": "TEXT",
    "url": "TEXT",
    "time_start": "TEXT",                # ISO 8601 UTC
    "time_end": "TEXT",
    "west": "REAL", "south": "REAL", "east": "REAL", "north": "REAL",
    "footprint": "TEXT",                 # CMR spatial extent, JSON
    "regions": "TEXT",                   # comma-separated region keys the granule intersects
    "windows": "TEXT",                   # comma-separated window labels
    "size_bytes": "INTEGER",             # expected size from the archive (may be approximate)
    "size_exact": "INTEGER",             # 1 if size_bytes is exact bytes, 0 if estimated
    "checksum_algo": "TEXT",             # archive-published checksum, if any
    "checksum_expected": "TEXT",
    "request": "TEXT",                   # JSON: subset request for Copernicus Marine rows
    "local_path": "TEXT",                # relative to data_root
    "status": "TEXT DEFAULT 'planned'",
    "bytes_on_disk": "INTEGER",
    "server_bytes": "INTEGER",           # Content-Length the server sent (authoritative file size)
    "size_note": "TEXT",                 # e.g. archive metadata size differs from the served file
    "sha256": "TEXT",
    "checksum_verified": "INTEGER",      # 1 matched the archive checksum, 0 none published, -1 mismatch
    "attempts": "INTEGER DEFAULT 0",
    "last_error": "TEXT",
    "download_seconds": "REAL",
    "downloaded_at": "TEXT",
    "inventoried_at": "TEXT",
}

TRANSFERS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "run_id": "TEXT",
    "granule_key": "TEXT",
    "product": "TEXT",
    "source": "TEXT",
    "host": "TEXT",
    "bytes": "INTEGER",
    "seconds": "REAL",
    "started_at": "TEXT",
    "ok": "INTEGER",
}

RUNS = {
    "run_id": "TEXT PRIMARY KEY",
    "host": "TEXT",
    "products": "TEXT",
    "connections": "INTEGER",
    "started_at": "TEXT",
    "finished_at": "TEXT",
    "files_ok": "INTEGER",
    "files_failed": "INTEGER",
    "bytes": "INTEGER",
    "wall_seconds": "REAL",
}

# Columns an inventory may overwrite. Download state is never touched by an inventory.
INVENTORY_COLUMNS = [
    "product", "kind", "family", "source", "short_name", "version", "provider", "concept_id",
    "granule_ur", "filename", "url", "time_start", "time_end", "west", "south", "east", "north",
    "footprint", "size_bytes", "size_exact", "checksum_algo", "checksum_expected", "request",
    "local_path",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120, check_same_thread=False)
        # One connection is shared by the download threads; sqlite3 connections are not safe for
        # concurrent use, so every statement and commit goes through this lock.
        self._lock = threading.RLock()
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")  # WAL is unsafe on network filesystems
        self._create("granules", COLUMNS)
        self._create("transfers", TRANSFERS)
        self._create("runs", RUNS)
        self.db.execute("CREATE INDEX IF NOT EXISTS g_product ON granules(product, status)")
        self.db.execute("CREATE INDEX IF NOT EXISTS g_time ON granules(time_start)")
        self.db.commit()

    def _create(self, table: str, columns: dict[str, str]) -> None:
        cols = ", ".join(f"{k} {v}" for k, v in columns.items())
        self.db.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
        have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
        for k, v in columns.items():  # forward-compatible: add columns introduced later
            if k not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {k} {v.replace('PRIMARY KEY', '')}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self.db
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    # ------------------------------------------------------------ inventory
    def upsert_inventory(self, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
        """Insert new granules; refresh archive metadata of known ones; merge regions/windows."""
        new = updated = 0
        now = utcnow()
        with self.transaction() as db:
            for row in rows:
                old = db.execute("SELECT regions, windows FROM granules WHERE granule_key=?",
                                 (row["granule_key"],)).fetchone()
                regions = _merge_list(old["regions"] if old else "", row.get("regions", ""))
                windows = _merge_list(old["windows"] if old else "", row.get("windows", ""))
                values = {c: row.get(c) for c in INVENTORY_COLUMNS}
                for c in ("footprint", "request"):
                    if values[c] is not None and not isinstance(values[c], str):
                        values[c] = json.dumps(values[c], separators=(",", ":"))
                if old is None:
                    cols = ["granule_key", *INVENTORY_COLUMNS, "regions", "windows", "status", "inventoried_at"]
                    vals = [row["granule_key"], *values.values(), regions, windows, "planned", now]
                    db.execute(f"INSERT INTO granules ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
                    new += 1
                else:
                    sets = ", ".join(f"{c}=?" for c in INVENTORY_COLUMNS)
                    db.execute(
                        f"UPDATE granules SET {sets}, regions=?, windows=?, inventoried_at=? WHERE granule_key=?",
                        [*values.values(), regions, windows, now, row["granule_key"]],
                    )
                    updated += 1
        return new, updated

    # ------------------------------------------------------------- queries
    def rows(self, where: str = "1=1", params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(f"SELECT * FROM granules WHERE {where} ORDER BY product, time_start",
                                   params).fetchall()

    def update(self, key: str, **values: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in values)
        with self.transaction() as db:
            db.execute(f"UPDATE granules SET {sets} WHERE granule_key=?", (*values.values(), key))

    def add_transfer(self, **values: Any) -> None:
        cols = ", ".join(values)
        with self.transaction() as db:
            db.execute(f"INSERT INTO transfers ({cols}) VALUES ({','.join('?' * len(values))})", tuple(values.values()))

    def save_run(self, **values: Any) -> None:
        cols = ", ".join(values)
        with self.transaction() as db:
            db.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({','.join('?' * len(values))})",
                       tuple(values.values()))

    def summary(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT product, status, COUNT(*) AS n, SUM(size_bytes) AS expected_bytes,
                      SUM(COALESCE(bytes_on_disk, 0)) AS on_disk
               FROM granules GROUP BY product, status ORDER BY product, status"""
        ).fetchall()

    def export_csv(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cur = self.db.execute("SELECT * FROM granules ORDER BY product, time_start")
        names = [d[0] for d in cur.description]
        with open(dest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([n for n in names if n != "footprint"])
            for r in cur:
                w.writerow([r[n] for n in names if n != "footprint"])
        return dest

    def close(self) -> None:
        with self._lock:
            self.db.close()


def _merge_list(a: str | None, b: str | None) -> str:
    items = {x for x in (a or "").split(",") if x} | {x for x in (b or "").split(",") if x}
    return ",".join(sorted(items))
