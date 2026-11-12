import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oefm.manifest import Manifest


class ManifestThreadingTest(unittest.TestCase):
    def test_concurrent_updates_from_threads(self):
        # The download pool shares one connection; unsynchronised use raised
        # "cannot start a transaction within a transaction" on Ptolemy (job 116018).
        with tempfile.TemporaryDirectory() as d:
            man = Manifest(Path(d) / "m.sqlite")
            keys = [f"p:f{i}" for i in range(200)]
            with man.transaction() as db:
                for k in keys:
                    db.execute("INSERT INTO granules (granule_key, product, status) VALUES (?, 'p', 'planned')", (k,))

            def work(k):
                for _ in range(5):
                    man.update(k, status="downloading")
                    man.rows("granule_key=?", (k,))
                man.add_transfer(run_id="r", granule_key=k, product="p", bytes=1, seconds=1.0, ok=1)
                man.update(k, status="verified")

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(work, keys))
            self.assertEqual(len(man.rows("status='verified'")), len(keys))
            man.close()


if __name__ == "__main__":
    unittest.main()
