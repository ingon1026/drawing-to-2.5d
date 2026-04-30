"""Seed the motion library with our my_dance series + AD's bundled fair1 motions.

Usage:
    cd stickerbook
    python scripts/seed_library.py

Idempotent: already-existing names are skipped. Missing source BVHs are
warned and skipped (e.g., AD repo not present).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from config import AD_REPO_PATH, ROOT  # noqa: E402
from motion.library import MotionLibrary  # noqa: E402


# (BVH path under AD repo, library name, preset)
SEED = [
    ("examples/bvh/my_dance.bvh",          "dance_1",       "rokoko"),
    ("examples/bvh/my_dance_2.bvh",        "dance_2",       "rokoko"),
    ("examples/bvh/my_dance_3.bvh",        "dance_3",       "rokoko"),
    ("examples/bvh/fair1/dab.bvh",         "dab",           "fair1"),
    ("examples/bvh/fair1/wave_hello.bvh",  "wave_hello",    "fair1"),
    ("examples/bvh/fair1/jumping_jacks.bvh","jumping_jacks","fair1"),
    ("examples/bvh/fair1/zombie.bvh",      "zombie",        "fair1"),
    ("examples/bvh/fair1/jesse_dance.bvh", "jesse_dance",   "fair1"),
]


def main() -> int:
    library = MotionLibrary(
        library_dir=ROOT / "assets" / "motions" / "library",
        ad_repo_path=AD_REPO_PATH,
    )
    existing = set(library.list())
    n_added = n_skip = n_missing = 0
    for rel_bvh, name, preset in SEED:
        if name in existing:
            print(f"[seed] skip  {name} (already in library)")
            n_skip += 1
            continue
        src = AD_REPO_PATH / rel_bvh
        if not src.is_file():
            print(f"[seed] WARN  missing source: {src}")
            n_missing += 1
            continue
        actual = library.add(src, name=name, preset=preset)
        print(f"[seed] added {actual:15s} (preset={preset:6s}, src={rel_bvh})")
        n_added += 1
    print(
        f"[seed] done. added={n_added}, skipped={n_skip}, "
        f"missing_src={n_missing}, total_in_lib={len(library.list())}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
