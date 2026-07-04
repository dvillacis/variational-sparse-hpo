#!/usr/bin/env python
"""Download the real-world datasets used by Experiment 6 (expes_fb/expe6_real_world).

There are two acquisition paths:

  auto    rcv1 via scikit-learn (``fetch_rcv1``) and mnist via ``libsvmdata`` —
          both manage their own caches; this script just triggers a fetch.
  manual  LIBSVM binary files fetched from the LIBSVM datasets page into
          ``expes_fb/expe6_real_world/data/`` (``.bz2`` archives are decompressed).

URLs and target filenames are read directly from ``data_loaders.py`` so this
script and the loaders never drift apart. Total download for the full manual set
is ~790 MB (news20.binary and real-sim dominate).

Examples
--------
    python scripts/download_data.py                    # auto + full manual set
    python scripts/download_data.py --datasets phishing w8a
    python scripts/download_data.py --auto             # only rcv1 + mnist
    python scripts/download_data.py --manual           # only the LIBSVM files
    python scripts/download_data.py --list             # show datasets, exit
"""

import argparse
import bz2
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPE6 = REPO_ROOT / "expes_fb" / "expe6_real_world"
DATA_DIR = EXPE6 / "data"

# Import the canonical URL/filename tables from the loader (single source of truth).
sys.path.insert(0, str(EXPE6))
from data_loaders import _LIBSVM_URLS, _LIBSVM_FILENAMES  # noqa: E402

# Default manual LIBSVM files to fetch (distinct target files only; rcv1.binary is
# an alias of rcv1_train.binary and is skipped to avoid a duplicate download).
DEFAULT_MANUAL = ["phishing", "w8a", "real-sim", "rcv1_train.binary", "news20.binary"]

# Rough decompressed sizes for the --list view (MB, approximate).
APPROX_MB = {
    "phishing": 1, "w8a": 5, "real-sim": 90, "rcv1_train.binary": 40,
    "news20.binary": 130,
}


def _report(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r      {done / 1e6:7.1f} / {total_size / 1e6:7.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


def download_manual(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final = DATA_DIR / _LIBSVM_FILENAMES.get(name, name)
    if final.exists():
        print(f"  · {name}: already present at {final.name}")
        return True
    url = _LIBSVM_URLS.get(name)
    if url is None:
        print(f"  ✗ {name}: no URL known (skipping)")
        return False

    archive = DATA_DIR / url.rsplit("/", 1)[-1]
    print(f"  ↓ {name}: {url}")
    try:
        urllib.request.urlretrieve(url, archive, _report)
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001 - surface any network/IO error
        print(f"\n  ✗ {name}: download failed ({exc})")
        return False

    if archive.suffix == ".bz2":
        print(f"    decompressing {archive.name} -> {final.name}")
        with bz2.open(archive, "rb") as src, open(final, "wb") as dst:
            shutil.copyfileobj(src, dst)
        archive.unlink()
    elif archive != final:
        archive.rename(final)
    print(f"  ✓ {name}: {final}")
    return True


def download_auto():
    ok = True
    print("  ↓ rcv1 (scikit-learn fetch_rcv1)")
    try:
        from data_loaders import load_rcv1
        load_rcv1(DATA_DIR)
        print("  ✓ rcv1")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ rcv1: {exc}")
        ok = False
    print("  ↓ mnist (libsvmdata)")
    try:
        from data_loaders import load_libsvmdata_binary
        load_libsvmdata_binary("mnist")
        print("  ✓ mnist")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ mnist: {exc}")
        ok = False
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", metavar="NAME",
                   help=f"manual LIBSVM files to fetch (default: {DEFAULT_MANUAL})")
    p.add_argument("--auto", action="store_true", help="only run the auto path (rcv1, mnist)")
    p.add_argument("--manual", action="store_true", help="only run the manual LIBSVM path")
    p.add_argument("--list", action="store_true", help="list datasets and exit")
    args = p.parse_args(argv)

    if args.list:
        print("auto (library-managed cache):  rcv1, mnist")
        print("manual LIBSVM files -> expes_fb/expe6_real_world/data/")
        for name in DEFAULT_MANUAL:
            print(f"  {name:20s} ~{APPROX_MB.get(name, '?')} MB")
        return 0

    do_auto = not args.manual
    do_manual = not args.auto
    manual_set = args.datasets or DEFAULT_MANUAL

    ok = True
    if do_auto:
        print("== auto datasets ==")
        ok = download_auto() and ok
    if do_manual:
        print(f"== manual LIBSVM files -> {DATA_DIR} ==")
        for name in manual_set:
            ok = download_manual(name) and ok

    print("\ndone." if ok else "\ndone with errors (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
