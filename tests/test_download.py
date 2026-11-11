import hashlib
import tempfile
import unittest
from pathlib import Path

from oefm.download import DownloadError, sniff, verify


def row(**kw):
    base = {"size_exact": 0, "size_bytes": None, "checksum_algo": None, "checksum_expected": None}
    base.update(kw)
    return base


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hdf5_signature_and_md5(self):
        p = self.dir / "a.nc"
        data = b"\x89HDF\r\n\x1a\n" + b"x" * 100
        p.write_bytes(data)
        self.assertEqual(sniff(p), "hdf5/netcdf4")
        info = verify(row(checksum_algo="md5", checksum_expected=hashlib.md5(data).hexdigest(),
                          size_exact=1, size_bytes=len(data)), p)
        self.assertEqual(info["checksum_verified"], 1)
        self.assertEqual(info["sha256"], hashlib.sha256(data).hexdigest())

    def test_html_login_page_rejected(self):
        p = self.dir / "b.nc"
        p.write_bytes(b"<!DOCTYPE html><title>Earthdata Login</title>")
        with self.assertRaises(DownloadError) as cm:
            verify(row(), p)
        self.assertIn("web page", str(cm.exception))

    def test_checksum_mismatch(self):
        p = self.dir / "c.nc"
        p.write_bytes(b"CDF\x01" + b"0" * 10)
        with self.assertRaises(DownloadError):
            verify(row(checksum_algo="md5", checksum_expected="0" * 32), p)

    def test_size_mismatch(self):
        p = self.dir / "d.nc"
        p.write_bytes(b"CDF\x02" + b"0" * 10)
        with self.assertRaises(DownloadError):
            verify(row(size_exact=1, size_bytes=999), p)

    def test_stale_archive_size_with_server_length(self):
        p = self.dir / "e.nc"
        p.write_bytes(b"CDF\x02" + b"0" * 10)
        info = verify(row(size_exact=1, size_bytes=999), p, expected_len=14)
        self.assertIn("differs", info["size_note"])
        with self.assertRaises(DownloadError):
            verify(row(size_exact=1, size_bytes=999), p, expected_len=15)


if __name__ == "__main__":
    unittest.main()
