"""MotionLibrary — BVH 라이브러리 + AD config 자동 등록 + 활성 motion 상태.

Persistence: library_dir에 *.bvh 파일들로 카운터/목록 자동 복원.
auto naming은 motion_NNN, 사용자 지정 이름도 허용 (sanitize 후 충돌 시 _2/_3 suffix).
AD 측 파일 (examples/bvh, examples/config/motion, examples/config/retarget)도
같이 생성. retarget yaml은 my_dance.yaml의 복사 (joint 이름 호환 가정).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional


_AUTO_RE = re.compile(r"^motion_(\d{3})\.bvh$")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


_MOTION_YAML_TEMPLATE = """\
filepath: examples/bvh/{name}.bvh
start_frame_idx: 0
end_frame_idx: {end_idx}
groundplane_joint: LeftFoot
forward_perp_joint_vectors:
  - - LeftShoulder
    - RightShoulder
  - - LeftThigh
    - RightThigh
scale: 0.025
up: +y
"""


class MotionLibrary:
    def __init__(self, library_dir: Path, ad_repo_path: Path) -> None:
        self._library_dir = Path(library_dir)
        self._ad_repo = Path(ad_repo_path)
        self._library_dir.mkdir(parents=True, exist_ok=True)
        self._active: Optional[str] = None

    def list(self) -> List[str]:
        names: List[str] = []
        for p in sorted(self._library_dir.iterdir()):
            if p.suffix == ".bvh":
                names.append(p.stem)
        return names

    def add(self, bvh_path: Path, name: Optional[str] = None) -> str:
        """Register a BVH into the library + AD examples/.

        Args:
            bvh_path: source BVH file
            name: user-supplied motion name (e.g. "dab"). If None, auto-assign
                  motion_NNN. If supplied name collides with existing entry,
                  suffix "_2", "_3", ... until unique.
        Returns: assigned name (may differ from `name` if conflict-suffixed).
        """
        bvh_path = Path(bvh_path)
        assigned = self._resolve_name(name)

        # 1. copy original to library
        dst_local = self._library_dir / f"{assigned}.bvh"
        shutil.copyfile(bvh_path, dst_local)

        # 2. copy to AD examples/bvh
        ad_bvh = self._ad_repo / "examples" / "bvh" / f"{assigned}.bvh"
        ad_bvh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bvh_path, ad_bvh)

        # 3. write motion config yaml
        end_idx = self._count_bvh_frames(bvh_path) - 1
        ad_motion_yaml = (
            self._ad_repo / "examples" / "config" / "motion" / f"{assigned}.yaml"
        )
        ad_motion_yaml.parent.mkdir(parents=True, exist_ok=True)
        ad_motion_yaml.write_text(
            _MOTION_YAML_TEMPLATE.format(name=assigned, end_idx=max(end_idx, 0))
        )

        # 4. retarget yaml = copy of my_dance.yaml (same joint name set)
        src_retarget = (
            self._ad_repo / "examples" / "config" / "retarget" / "my_dance.yaml"
        )
        ad_retarget_yaml = (
            self._ad_repo / "examples" / "config" / "retarget" / f"{assigned}.yaml"
        )
        if src_retarget.is_file():
            shutil.copyfile(src_retarget, ad_retarget_yaml)
        # If my_dance.yaml is missing, AD subprocess falls back to default
        # fair1_ppf — joint mismatch will surface in the smoke test.

        return assigned

    def set_active(self, name: str) -> None:
        if name not in self.list():
            raise ValueError(f"motion not in library: {name}")
        self._active = name

    def active(self) -> Optional[str]:
        return self._active

    def get_by_index(self, idx: int) -> Optional[str]:
        names = self.list()
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None

    def _resolve_name(self, name: Optional[str]) -> str:
        """Return the final on-disk-safe name to use, avoiding collisions."""
        sanitized = self._sanitize(name) if name is not None else None
        if not sanitized:
            return self._next_auto_name()

        if not (self._library_dir / f"{sanitized}.bvh").exists():
            return sanitized
        # collision — suffix _2, _3, ...
        i = 2
        while True:
            candidate = f"{sanitized}_{i}"
            if not (self._library_dir / f"{candidate}.bvh").exists():
                return candidate
            i += 1

    def _next_auto_name(self) -> str:
        """Find the next motion_NNN slot, scanning existing motion_NNN files."""
        used: List[int] = []
        for p in self._library_dir.iterdir():
            m = _AUTO_RE.match(p.name)
            if m:
                used.append(int(m.group(1)))
        next_n = (max(used) + 1) if used else 1
        return f"motion_{next_n:03d}"

    @staticmethod
    def _sanitize(name: str) -> str:
        """Allow alphanumerics, hyphens, underscores. Replace others with _."""
        stripped = name.strip()
        if not stripped:
            return ""
        return _SANITIZE_RE.sub("_", stripped)

    @staticmethod
    def _count_bvh_frames(path: Path) -> int:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("Frames:"):
                try:
                    return int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    return 0
        return 0
