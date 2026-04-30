"""MotionLibrary — BVH 라이브러리 + AD config 자동 등록 + 활성 motion 상태.

Persistence: library_dir에 motion_NNN.bvh 파일들로 카운터/목록 자동 복원.
AD 측 파일 (examples/bvh, examples/config/motion, examples/config/retarget)도
같이 생성. retarget yaml은 my_dance.yaml의 복사 (joint 이름 호환 가정).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional


_NAME_RE = re.compile(r"^motion_(\d{3})\.bvh$")


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
            m = _NAME_RE.match(p.name)
            if m:
                names.append(f"motion_{m.group(1)}")
        return names

    def add(self, bvh_path: Path) -> str:
        bvh_path = Path(bvh_path)
        existing = self.list()
        next_n = len(existing) + 1
        name = f"motion_{next_n:03d}"

        # 1. copy original to library
        dst_local = self._library_dir / f"{name}.bvh"
        shutil.copyfile(bvh_path, dst_local)

        # 2. copy to AD examples/bvh
        ad_bvh = self._ad_repo / "examples" / "bvh" / f"{name}.bvh"
        ad_bvh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bvh_path, ad_bvh)

        # 3. write motion config yaml
        end_idx = self._count_bvh_frames(bvh_path) - 1
        ad_motion_yaml = (
            self._ad_repo / "examples" / "config" / "motion" / f"{name}.yaml"
        )
        ad_motion_yaml.parent.mkdir(parents=True, exist_ok=True)
        ad_motion_yaml.write_text(
            _MOTION_YAML_TEMPLATE.format(name=name, end_idx=max(end_idx, 0))
        )

        # 4. retarget yaml = copy of my_dance.yaml (same joint name set)
        src_retarget = (
            self._ad_repo / "examples" / "config" / "retarget" / "my_dance.yaml"
        )
        ad_retarget_yaml = (
            self._ad_repo / "examples" / "config" / "retarget" / f"{name}.yaml"
        )
        if src_retarget.is_file():
            shutil.copyfile(src_retarget, ad_retarget_yaml)
        # If my_dance.yaml is missing, AD subprocess falls back to default
        # fair1_ppf — joint mismatch will surface in the smoke test.

        return name

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
