"""targets.match_eval: joins target sets with the imagery counts; side files in eval/ are skipped."""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestMatchEval(unittest.TestCase):
    def test_skips_components_and_moana(self):
        from oefm.config import load_config
        from oefm.targets import eval_dir, match_eval
        from oefm.tiling import tiles_root

        old = os.environ.get("OEFM_DATA_ROOT")
        with tempfile.TemporaryDirectory() as d:
            os.environ["OEFM_DATA_ROOT"] = d
            try:
                cfg = load_config()
                ed = eval_dir(cfg)
                ed.mkdir(parents=True)
                t = pd.DataFrame({"sample_id": ["a", "b", "c"], "dataset": "sfer", "lat": 25.0, "lon": -81.0,
                                  "time_utc": "2024-05-01T12:00:00Z", "label_split": ["train", "train", "test"],
                                  "chl": [0.1, 0.2, 0.3]})
                t.to_csv(ed / "sfer.csv.gz", index=False)
                pd.DataFrame({"component": [1], "x": [0.5]}).to_csv(ed / "edna_16S_nmf_k5_components.csv.gz", index=False)
                pd.DataFrame({"sample_id": ["a"], "moana_x": [1.0]}).to_csv(ed / "moana_matchups.csv.gz", index=False)
                idx = tiles_root(cfg) / "index"
                idx.mkdir(parents=True)
                pd.DataFrame({"dataset": "sfer", "sample_id": ["a", "b", "c"], "time_utc": "2024-05-01T12:00:00Z",
                              "tile_id": ["T1", "T1", None], "causal_PACE": [1, 0, 0], "causal_MODIS": [2, 0, 0]}
                             ).to_csv(idx / "eval_coverage.csv", index=False)
                logs = []
                match_eval(cfg, logs.append)
                out = sorted(p.name for p in (ed / "matched").glob("*.csv.gz"))
                self.assertEqual(out, ["sfer.csv.gz"])
                s = pd.read_csv(ed / "matched" / "summary.csv")
                self.assertEqual(int(s.loc[s["label_split"] == "train", "usable_optical_any"].iloc[0]), 1)
            finally:
                if old is None:
                    os.environ.pop("OEFM_DATA_ROOT", None)
                else:
                    os.environ["OEFM_DATA_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
