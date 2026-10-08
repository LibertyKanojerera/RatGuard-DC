"""RatGuard DC command line.

    python run.py download      # pull real DC data (about 10-30 minutes the first time)
    python run.py types         # list every 311 service type per year, marking the ones used
    python run.py build         # block group x month panel + features
    python run.py analyze       # models, lag test, equity audit, figures, app data
    python run.py all           # download + build + analyze

    python run.py synthetic     # FAKE data in the same format, to test the code offline
    python run.py all --synthetic
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="RatGuard DC pipeline")
    ap.add_argument("step", choices=["download", "types", "build", "analyze", "all", "synthetic"])
    ap.add_argument("--synthetic", action="store_true", help="use data/raw_synthetic (fake data) instead of data/raw")
    ap.add_argument("--years", type=int, nargs="*", help="only these 311 years (download step)")
    ap.add_argument("--force", action="store_true", help="re-download files that are already cached")
    a = ap.parse_args()
    t0 = time.time()

    if a.step == "synthetic":
        from ratguard import synthetic
        synthetic.generate()
        if not a.synthetic:
            print("Next: python run.py build --synthetic && python run.py analyze")
        return
    if a.step == "types":
        from ratguard import download
        download.show_types()
        return
    if a.step in ("download", "all") and not a.synthetic:
        from ratguard import download
        download.run(years=a.years, force=a.force)
    if a.step == "all" and a.synthetic:
        from ratguard import synthetic
        synthetic.generate()
    if a.step in ("build", "all"):
        from ratguard import build
        build.run(synthetic=a.synthetic)
    if a.step in ("analyze", "all"):
        from ratguard import analyze
        analyze.run()
    print(f"Finished in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
