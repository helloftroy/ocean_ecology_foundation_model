"""Command line: python -m oefm <command> [options]

  regions     print the pilot regions with their areas
  windows     print the merged satellite and physical time windows
  catalog     check collections/datasets, versions and archive time ranges (no login needed)
  inventory   dry-run inventory: search archives, fill the manifest, report counts and sizes
  download    resumable download of planned files (smoke test: --per-product 1)
  verify      re-verify files on disk against the manifest (size, signature, checksum)
  inspect     structure, wavelengths, flags, mask scenarios and maps for sample files
  index       build the observation index and multisensor coincidence tables
  overlap     compare closest-in-time PACE vs OLCI/MODIS/VIIRS swaths
  status      manifest summary by product and status
  report      write the pilot acquisition report (markdown) from everything above
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m oefm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config YAML (default config/pilot.yaml or $OEFM_CONFIG)")
    p.add_argument("--today", default=None, help="override 'today' (YYYY-MM-DD) for reproducible windows")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("regions")
    sub.add_parser("windows")
    c = sub.add_parser("catalog")
    c.add_argument("--products", nargs="*")

    i = sub.add_parser("inventory")
    i.add_argument("--products", nargs="*", help="product keys, families (PACE, OLCI...), 'satellite' or 'physical'")
    i.add_argument("--no-size-estimates", action="store_true", help="skip Copernicus Marine dry-run size estimates")

    for name in ("download", "verify"):
        d = sub.add_parser(name)
        d.add_argument("--products", nargs="*")
        d.add_argument("--start", help="only files ending on/after this date (YYYY-MM-DD)")
        d.add_argument("--end", help="only files starting on/before this date")
        d.add_argument("--region", help="only files intersecting this region")
        d.add_argument("--per-product", type=int, default=0, help="only the first N files of each product")
        d.add_argument("--max-files", type=int, default=0)
        d.add_argument("--connections", type=int, default=8)
        d.add_argument("--retries", type=int, default=5)
        d.add_argument("--retry-sleep", type=float, default=20.0)
        d.add_argument("--timeout", type=float, default=600.0, help="seconds without data before a retry")
        d.add_argument("--dry-run", action="store_true")
        d.add_argument("--recheck", action="store_true", help="include files already verified")

    s = sub.add_parser("inspect")
    s.add_argument("--products", nargs="*")
    s.add_argument("--per-product", type=int, default=2)
    s.add_argument("--date", help="prefer files from this date (default: the date most products share)")
    s.add_argument("--no-maps", action="store_true")

    x = sub.add_parser("index")
    x.add_argument("--workers", type=int, default=0)

    o = sub.add_parser("overlap")
    o.add_argument("--max-pairs", type=int, default=6)
    o.add_argument("--max-dt-hours", type=float, default=3.0)

    z = sub.add_parser("zenith", help="HISATZEN / sensor-zenith diagnostic (needs the index)")
    z.add_argument("--per-product", type=int, default=40, help="granules per product, spread over time")
    z.add_argument("--max-pairs", type=int, default=30, help="same-day pairs per sensor pair")
    z.add_argument("--max-dt-hours", type=float, default=3.0)
    z.add_argument("--workers", type=int, default=0)

    sub.add_parser("status")
    sub.add_parser("report")

    args = p.parse_args(argv)
    cfg = load_config(args.config, date.fromisoformat(args.today) if args.today else None)
    {"regions": cmd_regions, "windows": cmd_windows, "catalog": cmd_catalog, "inventory": cmd_inventory,
     "download": cmd_download, "verify": cmd_verify, "inspect": cmd_inspect, "index": cmd_index,
     "overlap": cmd_overlap, "zenith": cmd_zenith, "status": cmd_status, "report": cmd_report}[args.cmd](cfg, args)


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- regions / windows
def cmd_regions(cfg, args) -> None:
    regions = cfg.regions
    log(f"{'region':14s} {'west':>7s} {'south':>6s} {'east':>7s} {'north':>6s} {'W-E km':>7s} {'S-N km':>7s} {'area km2':>10s}")
    for r in [*regions.values(), cfg.domain()]:
        log(f"{r.key:14s} {r.west:7.2f} {r.south:6.2f} {r.east:7.2f} {r.north:6.2f} {r.width_km():7.0f} "
            f"{r.height_km():7.0f} {r.area_km2():10,.0f}")
    keys = list(regions)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ra, rb = regions[keys[a]], regions[keys[b]]
            w, e = max(ra.west, rb.west), min(ra.east, rb.east)
            s, n = max(ra.south, rb.south), min(ra.north, rb.north)
            if w < e and s < n:
                from .config import Region
                ov = Region("ov", "", w, s, e, n)
                log(f"overlap {ra.key} / {rb.key}: [{w}, {s}, {e}, {n}] {ov.area_km2():,.0f} km2")
    total = sum(r.area_km2() for r in regions.values())
    log(f"sum of region areas {total:,.0f} km2 (overlaps counted twice)")


def cmd_windows(cfg, args) -> None:
    for label, wins in (("satellite", cfg.satellite_windows()), ("physical", cfg.physical_windows())):
        log(f"{label} windows ({sum(w.days for w in wins):,} days):")
        for w in wins:
            log(f"  {w.start} .. {w.end} {w.days:5d} d  {', '.join(w.names)}")


# ----------------------------------------------------------------------------- catalog
def cmd_catalog(cfg, args) -> None:
    from .sources import cmems, nasa

    out = {}
    for key, prod in cfg.select_products(args.products).items():
        src = prod.get("source", "nasa_earthdata")
        if src == "nasa_earthdata":
            versions = nasa.describe_collection(prod["short_name"], provider=prod.get("provider"))
            want = [v for v in versions if v["version"] == prod.get("version")]
            others = sorted({v["version"] for v in versions} - {prod.get("version")})
            if want:
                v = want[0]
                log(f"{key:18s} {prod['short_name']} v{v['version']} {v['provider']} {v['concept_id']} "
                    f"{(v['begin'] or '')[:10]} .. {(v['end'] or 'ongoing')[:10]}"
                    + (f" | other versions in CMR: {', '.join(others)}" if others else ""))
            else:
                log(f"{key:18s} {prod['short_name']} version {prod.get('version')} NOT FOUND; CMR has {others}")
            out[key] = {"configured": prod, "cmr": versions}
        elif src == "copernicus_marine":
            infos = [cmems.dataset_info(d) for d in prod["datasets"]]
            for info in infos:
                log(f"{key:18s} {info['dataset_id']} v{info['version']} {info['start']} .. {info['end']} "
                    f"step {info['step_deg']} deg | {', '.join(info['variables'])}")
            out[key] = {"configured": prod, "datasets": infos}
        else:
            log(f"{key:18s} {prod.get('url')}")
            out[key] = {"configured": prod}
    dest = cfg.reports_dir / "catalog.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str))
    log(f"Catalog written to {dest}")


# ----------------------------------------------------------------------------- inventory
def cmd_inventory(cfg, args) -> None:
    from .manifest import Manifest
    from .sources import cmems, http, nasa

    man = Manifest(cfg.manifest_path)
    for key, prod in cfg.select_products(args.products).items():
        src = prod.get("source", "nasa_earthdata")
        log(f"Inventory {key} ({src})")
        if src == "nasa_earthdata":
            rows = nasa.inventory(prod, cfg, log)
        elif src == "copernicus_marine":
            rows = cmems.inventory(prod, cfg, log, estimate_sizes=not args.no_size_estimates)
        else:
            rows = http.inventory(prod, cfg, log)
        new, upd = man.upsert_inventory(rows)
        size = sum(r["size_bytes"] or 0 for r in rows)
        log(f"  -> {len(rows):,} files, {size / 1e9:,.1f} GB ({new:,} new, {upd:,} already in manifest)")
    csv_path = man.export_csv(cfg.data_root / "manifest" / "manifest.csv")
    man.close()
    cmd_status(cfg, args)
    log(f"Manifest: {cfg.manifest_path} (CSV copy {csv_path})")


def cmd_status(cfg, args) -> None:
    from .manifest import Manifest

    man = Manifest(cfg.manifest_path)
    from datetime import timedelta

    core = date.fromisoformat(str(cfg.raw["time"]["core"]["start"]))
    lead = core - timedelta(days=int(cfg.raw["time"].get("physical_lead_days", 0)))
    # "core" = the 2024-onward period (for physical products including its lead-in days)
    rows = man.db.execute(
        """SELECT product, COUNT(*) n, SUM(size_bytes) b,
                  SUM(CASE WHEN status='verified' THEN 1 ELSE 0 END) nv,
                  SUM(CASE WHEN status='verified' THEN bytes_on_disk ELSE 0 END) bv,
                  SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) nf,
                  SUM(CASE WHEN time_start IS NULL
                             OR substr(time_start,1,10) >= (CASE WHEN kind='physical' THEN ? ELSE ? END)
                           THEN size_bytes ELSE 0 END) b_core,
                  MIN(substr(time_start,1,10)) t0, MAX(substr(time_start,1,10)) t1
           FROM granules GROUP BY product ORDER BY product""", (lead.isoformat(), core.isoformat())).fetchall()
    log(f"{'product':18s} {'files':>7s} {'expected GB':>11s} {'core GB':>9s} {'hist GB':>8s} {'verified':>8s} "
        f"{'GB on disk':>10s} {'failed':>6s}  first .. last")
    tb = tv = 0
    for r in rows:
        b = (r["b"] or 0) / 1e9
        bc = (r["b_core"] or 0) / 1e9
        tb += b
        tv += (r["bv"] or 0) / 1e9
        log(f"{r['product']:18s} {r['n']:7,d} {b:11,.1f} {bc:9,.1f} {b - bc:8,.1f} {r['nv']:8,d} "
            f"{(r['bv'] or 0) / 1e9:10,.1f} {r['nf']:6,d}  {r['t0'] or '-'} .. {r['t1'] or '-'}")
    log(f"{'TOTAL':18s} {'':7s} {tb:11,.1f} {'':9s} {'':8s} {'':8s} {tv:10,.1f}")
    rate = best_rate(man)
    if rate:
        left = (tb - tv) * 1e9
        log(f"Best measured aggregate rate {rate / 1e6:,.1f} MB/s -> remaining {left / 1e12:,.2f} TB in "
            f"~{left / rate / 3600:,.1f} h")
    man.close()


def best_rate(man) -> float | None:
    r = man.db.execute("SELECT MAX(1.0 * bytes / wall_seconds) FROM runs WHERE bytes >= 1e9 AND wall_seconds > 0").fetchone()
    return r[0] if r and r[0] else None


# ----------------------------------------------------------------------------- download / verify
def cmd_download(cfg, args) -> None:
    from . import download

    prods = cfg.select_products(args.products)
    prods = {k: v for k, v in prods.items() if v.get("source") != "deferred"}
    download.run(cfg, prods, args, log)


def cmd_verify(cfg, args) -> None:
    from .download import DownloadError, verify
    from .manifest import Manifest
    from .download import select

    args.recheck = True
    man = Manifest(cfg.manifest_path)
    rows = select(man, list(cfg.select_products(args.products)), args)
    bad = Counter()
    for r in rows:
        path = cfg.data_root / r["local_path"]
        if not path.exists():
            if r["status"] == "verified":
                man.update(r["granule_key"], status="planned", last_error="missing on disk at verify")
                bad["missing"] += 1
            continue
        try:
            info = verify(r, path)
            man.update(r["granule_key"], status="verified", bytes_on_disk=info["bytes_on_disk"], sha256=info["sha256"],
                       checksum_verified=info["checksum_verified"], size_note=info["size_note"], last_error=None)
            bad["ok"] += 1
        except DownloadError as exc:
            man.update(r["granule_key"], status="failed", last_error=f"verify: {exc}")
            bad["failed"] += 1
    man.close()
    log(f"verify: {dict(bad)}")


# ----------------------------------------------------------------------------- inspect
def cmd_inspect(cfg, args) -> None:
    import csv

    from . import inspect as ins
    from .manifest import Manifest

    man = Manifest(cfg.manifest_path)
    prods = cfg.select_products(args.products)
    rows = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(prods))})", tuple(prods))]
    man.close()
    if not rows:
        log("No verified files yet: run the download smoke test first.")
        return
    by_day = defaultdict(set)
    for r in rows:
        if r["time_start"]:
            by_day[r["time_start"][:10]].add(r["product"])
    pref = args.date or (max(by_day, key=lambda d: (len(by_day[d]), d)) if by_day else None)
    log(f"Preferred date {pref} ({len(by_day.get(pref, []))} products have files that day)")
    picks: dict[str, list] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (0 if (r["time_start"] or "")[:10] == pref else 1, r["time_start"] or "")):
        if len(picks[r["product"]]) < args.per_product:
            picks[r["product"]].append(r)

    depth = ins.gebco_depth(cfg)
    log("GEBCO depth classes: " + ("yes" if depth else "no (GEBCO not downloaded yet)"))
    out_dir = cfg.reports_dir / "inspect"
    summary = []
    for key, chosen in picks.items():
        prod = prods[key]
        for r in chosen:
            path = cfg.data_root / r["local_path"]
            stem = Path(r["filename"]).stem
            try:
                ins.write_json(ins.structure(path), out_dir / key / f"{stem}.structure.json")
                if prod["kind"] == "satellite":
                    res = ins.inspect_l2(path, prod, cfg, depth)
                    maps = [] if args.no_maps else ins.map_l2(path, prod, cfg, out_dir / key / f"{stem}.png")
                    for reg, st in res["regions"].items():
                        summary.append({
                            "product": key, "file": r["filename"], "region": reg, "variable": res["variable"],
                            "n_bands": res["n_bands"], "wl_min": min(res["wavelengths_nm"] or [0]),
                            "wl_max": max(res["wavelengths_nm"] or [0]), "units": res["units"],
                            "n_pixels": st["n_pixels"], "along_scan_km": st.get("along_scan_km"),
                            "along_track_km": st.get("along_track_km"),
                            "nadir_scan_km": st.get("nadir_scan_km"), "nadir_track_km": st.get("nadir_track_km"),
                            "ref_nm": st["ref_wavelength_nm"], "ref_fill": st["ref_fill_fraction"],
                            "ref_negative": st["ref_negative_fraction"],
                            "cloud": st["flag_fraction"].get("CLDICE"), "land": st["flag_fraction"].get("LAND"),
                            "straylight": st["flag_fraction"].get("STRAYLIGHT"),
                            **{f"valid_{m}": v for m, v in st["mask_valid_fraction"].items()},
                            **{f"valid_{m}_{c}": v for m, cs in st.get("mask_valid_by_depth", {}).items()
                               for c, v in cs.items()},
                            "missing_bands": " ".join(f"{b:g}" for b in st["missing_bands"]),
                        })
                else:
                    if r["source"] == "http" and path.stat().st_size > 2e9:
                        res = {"file": r["filename"], "note": "large static grid: structure only"}
                    else:
                        res = ins.inspect_grid(path, prod, cfg)
                    maps = [] if args.no_maps else ins.map_grid(path, prod, cfg, out_dir / key / f"{stem}.png")
                ins.write_json(res, out_dir / key / f"{stem}.inspect.json")
                log(f"  {key:18s} {r['filename']} -> {len(maps)} map(s)")
            except Exception as exc:  # noqa: BLE001 - keep going, record it
                log(f"  {key:18s} {r['filename']} FAILED: {type(exc).__name__}: {exc}")
                ins.write_json({"file": r["filename"], "error": f"{type(exc).__name__}: {exc}"},
                               out_dir / key / f"{stem}.inspect.json")
    if summary:
        dest = out_dir / "l2_region_summary.csv"
        cols = sorted({k for s in summary for k in s}, key=lambda k: list(summary[0]).index(k) if k in summary[0] else 999)
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        log(f"Summary: {dest}")


# ----------------------------------------------------------------------------- index / overlap / report
def cmd_index(cfg, args) -> None:
    from . import index

    index.build(cfg, args.workers, log)


def cmd_overlap(cfg, args) -> None:
    from . import index

    dest = index.auto_matchups(cfg, args.max_pairs, args.max_dt_hours, log)
    log(f"Matchups: {dest}")


def cmd_zenith(cfg, args) -> None:
    from . import zenith

    dest = zenith.run(cfg, args.per_product, args.max_pairs, args.max_dt_hours, args.workers, log)
    log(f"Zenith summary: {dest}")


def cmd_report(cfg, args) -> None:
    from . import report

    dest = report.write(cfg, log)
    log(f"Report: {dest}")


if __name__ == "__main__":
    main(sys.argv[1:])
