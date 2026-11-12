"""Write the pilot acquisition report (markdown) from the manifest, benchmarks, QC and index outputs."""

from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

from .config import REPO_ROOT, Config
from .manifest import Manifest


def _gb(b) -> str:
    return f"{(b or 0) / 1e9:,.1f}"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write(cfg: Config, log=print) -> Path:
    man = Manifest(cfg.manifest_path)
    L: list[str] = []
    add = L.append
    add(f"# Pilot acquisition report: {cfg.raw.get('phase')}\n")
    add(f"Generated {date.today()} from `{cfg.manifest_path.relative_to(cfg.data_root.parent) if cfg.manifest_path.is_relative_to(cfg.data_root.parent) else cfg.manifest_path}`. "
        "Re-run `python -m oefm report` after each stage; every number below is recomputed.\n")

    # -------------------------------------------------------------- regions / windows
    add("## Regions and time windows\n")
    add("| region | bbox [W, S, E, N] | W-E km | S-N km | area km² |\n|---|---|---|---|---|")
    for r in [*cfg.regions.values(), cfg.domain()]:
        add(f"| {r.key} | [{r.west}, {r.south}, {r.east}, {r.north}] | {r.width_km():,.0f} | {r.height_km():,.0f} | {r.area_km2():,.0f} |")
    sw, pw = cfg.satellite_windows(), cfg.physical_windows()
    add(f"\nSatellite windows: {len(sw)} ({sum(w.days for w in sw):,} days). "
        f"Physical windows (with {cfg.raw['time'].get('physical_lead_days', 0)}-day lead): {len(pw)} ({sum(w.days for w in pw):,} days).\n")
    for w in sw:
        add(f"- {w.start} .. {w.end} ({w.days} d): {', '.join(w.names)}")
    add("")

    # -------------------------------------------------------------- 1 inventory
    add("## 1. Dataset inventory\n")
    add("| product | archive name | version | origin | files | first | last | expected GB |\n|---|---|---|---|---|---|---|---|")
    stats = {r["product"]: r for r in man.db.execute(
        """SELECT product, COUNT(*) n, SUM(size_bytes) b, MIN(substr(time_start,1,10)) t0,
                  MAX(substr(time_start,1,10)) t1, GROUP_CONCAT(DISTINCT version) v, GROUP_CONCAT(DISTINCT short_name) s
           FROM granules GROUP BY product""")}
    for key, p in cfg.products.items():
        s = stats.get(key)
        name = p.get("short_name") or ", ".join(p.get("datasets", [])) or p.get("url", "")
        add(f"| {key} | {name} | {(s['v'] if s else p.get('version')) or ''} | {p.get('origin', '')} | "
            f"{s['n'] if s else 0:,} | {(s['t0'] or 'static') if s else ''} | {(s['t1'] or 'static') if s else ''} | {_gb(s['b'] if s else 0)} |")
    add("\nDeferred (identified, not downloaded): " + "; ".join(
        f"{k} ({v.get('product_id')}, {v.get('origin')})" for k, v in cfg.raw.get("deferred", {}).items()) + "\n")
    cat = cfg.reports_dir / "catalog.json"
    if cat.exists():
        add(f"Live archive check: `{cat.name}` (from `python -m oefm catalog`).\n")

    # -------------------------------------------------------------- 2 storage
    add("## 2. Storage estimates\n")
    core = date.fromisoformat(str(cfg.raw["time"]["core"]["start"]))
    lead = core - timedelta(days=int(cfg.raw["time"].get("physical_lead_days", 0)))
    add("Raw = files as distributed (whole swaths / global grids). Regional = only pixels inside the regions, as "
        "float32 for every band plus flags and lat/lon (upper bound: assumes each granule covers its whole region).\n")
    add("| product | raw GB, 2024+ | raw GB, 2018-2023 | raw GB total | on disk GB | regional GB (est.) |\n|---|---|---|---|---|---|")
    tot = [0.0] * 5
    for key, p in cfg.products.items():
        cut = (lead if p["kind"] == "physical" else core).isoformat()
        r = man.db.execute(
            """SELECT SUM(CASE WHEN time_start IS NULL OR substr(time_start,1,10) >= ? THEN size_bytes ELSE 0 END) c,
                      SUM(size_bytes) t, SUM(CASE WHEN status='verified' THEN bytes_on_disk ELSE 0 END) d
               FROM granules WHERE product=?""", (cut, key)).fetchone()
        c, t, d = r["c"] or 0, r["t"] or 0, r["d"] or 0
        reg = regional_bytes(cfg, man, key, p, t)
        for i, v in enumerate((c, t - c, t, d, reg)):
            tot[i] += v
        add(f"| {key} | {_gb(c)} | {_gb(t - c)} | {_gb(t)} | {_gb(d)} | {_gb(reg)} |")
    add(f"| **total** | {_gb(tot[0])} | {_gb(tot[1])} | {_gb(tot[2])} | {_gb(tot[3])} | {_gb(tot[4])} |\n")

    # -------------------------------------------------------------- 3 pipeline
    add("## 3. Acquisition pipeline\n")
    add("Code: `oefm/` (run as `python -m oefm <command>`); Ptolemy jobs: `cluster/`. Manifest: "
        f"`{cfg.manifest_path.name}` (SQLite) with a CSV copy `manifest.csv`. Status counts:\n")
    add("| product | status | files | GB expected | GB on disk |\n|---|---|---|---|---|")
    for r in man.summary():
        add(f"| {r['product']} | {r['status']} | {r['n']:,} | {_gb(r['expected_bytes'])} | {_gb(r['on_disk'])} |")
    fails = man.db.execute("SELECT product, last_error, COUNT(*) n FROM granules WHERE status='failed' "
                           "GROUP BY product, last_error ORDER BY n DESC LIMIT 10").fetchall()
    if fails:
        add("\nMost common failures:\n")
        for f in fails:
            add(f"- {f['product']} ({f['n']}): `{(f['last_error'] or '')[:160]}`")
    add("")

    # -------------------------------------------------------------- 4 QC
    add("## 4. Quality-control summary\n")
    qc = _read_csv(cfg.reports_dir / "inspect" / "l2_region_summary.csv")
    if qc:
        add("Per sample granule and region. Valid fractions are the share of in-region pixels kept by each candidate "
            "mask (nothing was removed from the files). Depth classes use GEBCO when it is downloaded.\n")
        add("| product | region | bands | nm | pixel km in region (scan x track) | at nadir | cloud | straylight | neg. ref | valid minimal | "
            "valid L3 w/o straylight | valid L3 default | L3 default, 0-20 m |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for q in qc:
            add(f"| {q['product']} | {q['region']} | {q['n_bands']} | {float(q['wl_min']):.0f}-{float(q['wl_max']):.0f} | "
                f"{_f(q.get('along_scan_km'))} x {_f(q.get('along_track_km'))} | "
                f"{_f(q.get('nadir_scan_km'))} x {_f(q.get('nadir_track_km'))} | {_f(q.get('cloud'))} | "
                f"{_f(q.get('straylight'))} | {_f(q.get('ref_negative'))} | {_f(q.get('valid_minimal'))} | "
                f"{_f(q.get('valid_l3_no_straylight'))} | {_f(q.get('valid_l3_default'))} | "
                f"{_f(q.get('valid_l3_default_coastal_0_20m'))} |")
        figs = sorted((cfg.reports_dir / "inspect").rglob("*.png"))
        add("\nThe highest per-flag fractions per product (all flags are in the inspect JSON):\n")
        for js in sorted((cfg.reports_dir / "inspect").glob("*/*.inspect.json")):
            d = json.loads(js.read_text())
            for reg, st in (d.get("regions") or {}).items():
                top = sorted(st["flag_fraction"].items(), key=lambda kv: -kv[1])[:6]
                add(f"- {d['product']} / {reg}: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        add(f"\nDiagnostic maps ({len(figs)}): `{(cfg.reports_dir / 'inspect').relative_to(cfg.data_root)}/<product>/`.\n")
        sf = list((cfg.reports_dir / "inspect" / "pace_oci_sfrefl").glob("*.inspect.json"))
        if sf:
            d = json.loads(sf[0].read_text())
            if "sargassum_bands" in d:
                add("PACE SFREFL bands near floating-algae index wavelengths: " + ", ".join(
                    f"{k} -> {v['nearest_nm']:g} nm" for k, v in d["sargassum_bands"].items() if isinstance(v, dict)) + ".\n")
    else:
        add("_Not run yet (`python -m oefm inspect`)._\n")
    mu = cfg.reports_dir / "overlap" / "matchups.json"
    if mu.exists():
        add("Sensor matchups (0.02° bins, minimal mask):\n")
        add("| region | a | b | Δt h | band a/b nm | cells | r | median b/a |\n|---|---|---|---|---|---|---|---|")
        for m in json.loads(mu.read_text()):
            for b in m.get("bands", []):
                add(f"| {m['region']} | {m['a']} | {m['b']} | {m.get('dt_h')} | {b['a_nm']:.0f}/{b['b_nm']:.0f} | "
                    f"{b['n_cells']} | {b.get('pearson_r', '')} | {b.get('median_ratio_b_over_a', '')} |")
        add("")
    zs = cfg.reports_dir / "zenith" / "zenith_summary.json"
    if zs.exists():
        z = json.loads(zs.read_text())
        add("### HISATZEN / sensor-zenith diagnostic\n")
        add("Pilot policy: HISATZEN is kept as a lower-confidence tag, not masked (mask `l3_no_hisatzen` = L3 "
            "default minus HISATZEN). Sensor zenith is reconstructed from each file's geolocation and orbit "
            "altitude; the HISATZEN flag edge should fall at l2gen's threshold if decoding and geometry agree. "
            f"Figures and CSVs: `{zs.parent}`.\n")
        add("Decode check:\n")
        add("| product | granules | HISATZEN bit | l2gen satzen / masksatzen | flag = (zen ≥ thr), median (min) | "
            "zen at flag edge | max zen |\n|---|---|---|---|---|---|---|")
        for d in z.get("decode", []):
            add(f"| {d['product']} | {d['granules']} | {d['hisatzen_bits']} | {d['l2gen_satzen']} / "
                f"{d['l2gen_masksatzen']} | {_f(d['flag_matches_angle_median'])} ({_f(d['flag_matches_angle_min'])}) | "
                f"{_f(d['flag_edge_zen_median'])} ({d['granules_with_flag_edge']} granules) | {_f(d['zen_max'])} |")
        add("\nValid water pixels per region (share of water pixels; `x` = relaxed / L3 default):\n")
        add("| product | region | granules | HISATZEN share | L3 default | L3 without HISATZEN | minimal | x | "
            "granules empty under L3 only |\n|---|---|---|---|---|---|---|---|---|")
        for c in z.get("coverage", []):
            add(f"| {c['product']} | {c['region']} | {c['granules']} | {_f(c['hisatzen_fraction'])} | "
                f"{_f(c['valid_l3_default'])} | {_f(c['valid_l3_no_hisatzen'])} | {_f(c['valid_minimal'])} | "
                f"{_f(c['recovered_x'])} | {c['granules_l3_empty_relaxed_not']} |")
        eff = [e for e in z.get("angle_effect_by_target", []) if e["target_zen_class"] in ("45-60", "60-90")]
        if eff:
            add("\nAngle effect from same-day matchups against a reference sensor viewing at < 45°: the "
                "target/reference ratio in each target-zenith class divided by the same ratio when the target "
                "views at < 45° (1.0 = no angle dependence; median and range over reference sensors):\n")
            add("| target | band nm | target zenith | references | pairs | cells | ratio vs < 45° (range) |\n"
                "|---|---|---|---|---|---|---|")
            for e in eff:
                add(f"| {e['target']} | {e['band']} | {e['target_zen_class']} | {e['references']} | {e['pairs']} | "
                    f"{e['cells']:,} | {_f(e['ratio_vs_low_median'])} ({_f(e['ratio_vs_low_min'])}-"
                    f"{_f(e['ratio_vs_low_max'])}) |")
        add("")

    # -------------------------------------------------------------- 5 benchmarks
    add("## 5. Transfer benchmarks\n")
    runs = man.db.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
    if runs:
        add("| run | host | connections | files ok/failed | GB | minutes | MB/s aggregate |\n|---|---|---|---|---|---|---|")
        for r in runs:
            add(f"| {r['run_id']} | {r['host']} | {r['connections']} | {r['files_ok']}/{r['files_failed']} | "
                f"{_gb(r['bytes'])} | {(r['wall_seconds'] or 0) / 60:,.1f} | {(r['bytes'] or 0) / max(r['wall_seconds'] or 1, 1) / 1e6:,.1f} |")
        per = {r["source"]: r for r in man.db.execute(
            """SELECT source, COUNT(*) n, SUM(bytes) b, SUM(seconds) s FROM transfers GROUP BY source""")}
        left = {r["source"]: r["b"] or 0 for r in man.db.execute(
            "SELECT source, SUM(size_bytes) b FROM granules WHERE status != 'verified' GROUP BY source")}
        conns = runs[-1]["connections"] or 8
        add(f"\nProjection: remaining bytes per source / (measured per-connection rate x connections; "
            f"{conns} for NASA and HTTP, 2 for Copernicus Marine). Sources are fetched in parallel, so the "
            f"pilot takes about as long as the slowest source. Servers may throttle at higher concurrency; "
            f"the next large run's aggregate rate is the better number.\n")
        add("| source | files fetched | GB fetched | per-connection MB/s | remaining GB | projected hours |\n|---|---|---|---|---|---|")
        worst = 0.0
        for src in sorted(set(per) | set(left)):
            r = per.get(src)
            rate = (r["b"] or 0) / r["s"] if r and r["s"] else 0
            c = min(2, conns) if src == "copernicus_marine" else conns
            hours = left.get(src, 0) / (rate * c) / 3600 if rate else float("nan")
            worst = max(worst, hours) if hours == hours else worst
            add(f"| {src} | {r['n'] if r else 0:,} | {_gb(r['b'] if r else 0)} | {rate / 1e6:,.1f} | "
                f"{_gb(left.get(src, 0))} | {hours:,.1f} |")
        total_left = sum(left.values())
        add(f"\nRemaining {total_left / 1e12:,.2f} TB; projected about {worst:,.1f} hours "
            f"({worst / 24:,.1f} days) if the per-connection rates hold.\n")
    else:
        add("_No download runs recorded yet._\n")

    # -------------------------------------------------------------- 6 readiness
    add("## 6. Data-readiness assessment\n")
    add("### Observation index\n")
    counts = _read_csv(cfg.data_root / "index" / "counts_by_sensor_region_month.csv")
    if counts:
        agg: dict[tuple, int] = {}
        for c in counts:
            k = (c["family"], c["region"], c["year"])
            agg[k] = agg.get(k, 0) + int(c["observations"])
        add("| family | region | year | observations |\n|---|---|---|---|")
        for (f, r, y), n in sorted(agg.items()):
            add(f"| {f} | {r} | {y} | {n:,} |")
        co = _read_csv(cfg.data_root / "index" / "multisensor_coincidences.csv")
        if co:
            pace = [c for c in co if "PACE" in c["families"]]
            add(f"\n{len(co):,} region-days with valid data from 2+ instrument families; {len(pace):,} include PACE.\n")
    else:
        add("_Index not built yet (`python -m oefm index`)._\n")
    notes = REPO_ROOT / "docs" / "readiness_notes.md"
    if notes.exists():
        add(notes.read_text())
    man.close()

    dest = cfg.reports_dir / "pilot_acquisition_report.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(L) + "\n")
    return dest


def _f(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return ""


def regional_bytes(cfg: Config, man: Manifest, key: str, p: dict, raw_total: float) -> float:
    """Bytes needed to keep only the regions (float32 bands + flags + lat/lon)."""
    if p["kind"] == "satellite":
        res_km = p.get("nominal_resolution_m", 1000) / 1000
        per_px = 4 * (p.get("nominal_bands", 10) + 1 + 2)
        total = 0.0
        for (regions,) in man.db.execute("SELECT regions FROM granules WHERE product=?", (key,)):
            for r in (regions or "").split(","):
                if r in cfg.regions:
                    total += cfg.regions[r].area_km2() / res_km**2 * per_px
        return total
    if p.get("source") == "copernicus_marine":
        return raw_total  # already regional subsets
    # Global grids: the domain's share of the globe (compressed sizes, rough).
    return raw_total * cfg.domain().area_km2() / 510.1e6
