"""
check_structure.py — Run this before pushing to GitHub, to make sure
every file is in the right place and named correctly.

Usage:
    python check_structure.py
"""

import os
import sys

REQUIRED_ROOT_FILES = [
    "app.py",
    "pricing_bs.py",
    "pricing_floating.py",
    "pricing_heston.py",
    "pricing_adaptive.py",
    "requirements.txt",
]

REQUIRED_PAGE_FILES = [
    "00_home.py",
    "01_bs_barrier.py",
    "02_mc_naive.py",
    "03_brownian_bridge.py",
    "04_mc_naive_floating.py",
    "05_mc_bb_floating.py",
    "06_npi_floating.py",
    "07_comparison.py",
    "08_adaptive_npi.py",
    "09_heston.py",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    problems = []

    print(f"Checking project structure in: {here}\n")

    for fname in REQUIRED_ROOT_FILES:
        path = os.path.join(here, fname)
        if os.path.isfile(path):
            print(f"  [OK]      {fname}")
        else:
            print(f"  [MANCA]   {fname}")
            problems.append(fname)

    pages_dir = os.path.join(here, "pages")
    if not os.path.isdir(pages_dir):
        print(f"\n  [MANCA]   cartella pages/  <-- deve esistere, con questo nome esatto")
        problems.append("pages/")
    else:
        print(f"\n  [OK]      cartella pages/")
        for fname in REQUIRED_PAGE_FILES:
            path = os.path.join(pages_dir, fname)
            if os.path.isfile(path):
                print(f"  [OK]      pages/{fname}")
            else:
                print(f"  [MANCA]   pages/{fname}")
                problems.append(f"pages/{fname}")

        # Check for stray files with spaces instead of underscores
        # (common mistake when renaming downloaded files).
        for actual in os.listdir(pages_dir):
            if " " in actual:
                print(f"  [ATTENZIONE] pages/{actual}  <-- contiene uno spazio, "
                      f"rinominalo con underscore (_)")
                problems.append(f"pages/{actual} (spazio nel nome)")

    print()
    if problems:
        print(f"Trovati {len(problems)} problema/i. Correggi prima di caricare su GitHub:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("Tutto a posto! La struttura è pronta per essere caricata su GitHub.")
        sys.exit(0)


if __name__ == "__main__":
    main()
