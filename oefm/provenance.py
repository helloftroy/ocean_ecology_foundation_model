"""Processed-data layout and the processing manifest (step 2.0).

    data/raw/                    immutable downloads, shared by every processed dataset
    data/processed/regional/     Florida/Gulf pilot (this phase)
        processing.sqlite        what was made, from which raw files, by which code, how
        physical/<product>/...   regional crops of the physical fields (step 2.2)
        tiles/<version>/         tile grid, per-granule tile stores, observation index (step 2.3)
    data/processed/global/       created now, left empty until global processing starts

Every output gets one row in `outputs`: step, product, the raw source files (manifest key, file
name, sha256), the code version (git commit, dirty flag, processing version), the parameters
(subset, grid, masks) and the transformations applied. Only the parent process writes the
database, so worker processes never touch SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS outputs (
    output_path TEXT PRIMARY KEY,      -- relative to the data root
    step TEXT NOT NULL,                -- crop | tiles | grid | examples
    product TEXT,
    status TEXT NOT NULL,              -- done | empty | failed
    sources TEXT,                      -- JSON list of {granule_key, filename, sha256}
    code_version TEXT,                 -- JSON {git_commit, git_dirty, processing_version, oefm}
    params TEXT,                       -- JSON: subset, grid, masks, ...
    transformations TEXT,              -- JSON list of short descriptions
    n_records INTEGER,
    bytes INTEGER,
    error TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS tile_obs (
    product TEXT, family TEXT, granule_key TEXT, filename TEXT, output_path TEXT, row INTEGER,
    tile_id TEXT, time_utc TEXT, valid_fraction REAL, strict_fraction REAL, cover_fraction REAL,
    hisatzen_fraction REAL, senz_mean REAL, senz_max REAL, solz REAL, n_pixels INTEGER,
    valid_fraction_300m REAL,
    PRIMARY KEY (granule_key, tile_id)
);
CREATE INDEX IF NOT EXISTS tile_obs_tile ON tile_obs (tile_id, time_utc);
"""


def processed_root(cfg: Config) -> Path:
    return cfg.data_root / "processed"


def regional_root(cfg: Config) -> Path:
    return processed_root(cfg) / "regional"


def processing_version(cfg: Config) -> str:
    return str(cfg.raw.get("processing", {}).get("version", "v1"))


def code_version(cfg: Config) -> dict[str, Any]:
    from . import __version__

    out: dict[str, Any] = {"processing_version": processing_version(cfg), "oefm": __version__}
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=20).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        out["git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        out["git_commit"], out["git_dirty"] = None, None
    return out


def make_layout(cfg: Config, log=print) -> dict[str, Path]:
    """Create the processed directories, their READMEs and the empty processing manifest."""
    reg, glo = regional_root(cfg), processed_root(cfg) / "global"
    for d in (reg, reg / "physical", reg / "tiles", glo):
        d.mkdir(parents=True, exist_ok=True)
    readme_reg = reg / "README.md"
    if not readme_reg.exists():
        readme_reg.write_text(
            "# Florida/Gulf pilot: processed data\n\n"
            "Built from `../../raw/` by `python -m oefm` (steps 2.2-2.4 in cluster_instructions.md).\n"
            "Nothing here is an original download; everything can be regenerated from raw/ and the\n"
            "code version recorded in `processing.sqlite` (table `outputs`).\n\n"
            "- `physical/<product>/<year>/`: regional crops of the physical fields, original variables,\n"
            "  units, encoding and metadata, plus `oefm_*` provenance attributes.\n"
            "- `tiles/<version>/`: the tile grid (`grid/`), one zipped Zarr store per satellite granule\n"
            "  (`<product>/<year>/<granule>.zarr.zip`) and the observation index (`index/`).\n")
    readme_glo = glo / "README.md"
    if not readme_glo.exists():
        readme_glo.write_text("# Global processed data\n\nIntentionally empty until global processing begins. "
                              "It will reference the same `../../raw/` archive.\n")
    man = ProcessingManifest(cfg)
    man.close()
    log(f"Processed layout ready: {reg} and {glo} (manifest {man.path})")
    return {"regional": reg, "global": glo, "manifest": man.path}


class ProcessingManifest:
    def __init__(self, cfg: Config) -> None:
        self.root = cfg.data_root
        self.path = regional_root(cfg) / "processing.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def done(self, step: str) -> set[str]:
        """Output paths of this step already finished (done or empty)."""
        return {r[0] for r in self.db.execute(
            "SELECT output_path FROM outputs WHERE step=? AND status IN ('done', 'empty')", (step,))}

    def record(self, output: Path | str, step: str, product: str | None, status: str, sources: list[dict],
               code: dict, params: dict, transformations: list[str], n_records: int | None = None,
               error: str | None = None, tile_rows: list[dict] | None = None) -> None:
        rel = str(Path(output).relative_to(self.root)) if Path(output).is_absolute() else str(output)
        size = Path(output).stat().st_size if Path(output).is_absolute() and Path(output).exists() else None
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO outputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rel, step, product, status, json.dumps(sources), json.dumps(code), json.dumps(params, default=str),
                 json.dumps(transformations), n_records, size, error, datetime.now(timezone.utc).isoformat()))
            if tile_rows:
                cols = ["product", "family", "granule_key", "filename", "output_path", "row", "tile_id", "time_utc",
                        "valid_fraction", "strict_fraction", "cover_fraction", "hisatzen_fraction", "senz_mean",
                        "senz_max", "solz", "n_pixels", "valid_fraction_300m"]
                self.db.executemany(
                    f"INSERT OR REPLACE INTO tile_obs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [tuple(r.get(c) if c != "output_path" else rel for c in cols) for r in tile_rows])

    def prune_version(self, version: str) -> dict[str, int]:
        """Delete the records of one tiles/<version>/ (outputs and tile observations), then VACUUM.
        The files themselves are not touched (delete the folder separately)."""
        like = f"processed/regional/tiles/{version}/%"
        with self.db:
            n_obs = self.db.execute("DELETE FROM tile_obs WHERE output_path LIKE ?", (like,)).rowcount
            n_out = self.db.execute("DELETE FROM outputs WHERE output_path LIKE ?", (like,)).rowcount
        self.db.execute("VACUUM")
        return {"tile_obs": n_obs, "outputs": n_out}

    def summary(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT step, product, status, COUNT(*) n, SUM(bytes) b, SUM(n_records) r "
                               "FROM outputs GROUP BY step, product, status ORDER BY step, product, status").fetchall()
