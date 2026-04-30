"""Unit tests for MotionLibrary."""
from __future__ import annotations

from pathlib import Path

import pytest

from motion.library import MotionLibrary


@pytest.fixture
def fake_ad_repo(tmp_path: Path) -> Path:
    """Mimic AnimatedDrawings repo dirs and seed retarget my_dance.yaml."""
    ad_repo = tmp_path / "ad_repo"
    (ad_repo / "examples" / "bvh").mkdir(parents=True)
    (ad_repo / "examples" / "config" / "motion").mkdir(parents=True)
    rt_dir = ad_repo / "examples" / "config" / "retarget"
    rt_dir.mkdir(parents=True)
    (rt_dir / "my_dance.yaml").write_text("char_starting_location: [0,0,-0.5]\n")
    return ad_repo


def make_dummy_bvh(path: Path) -> Path:
    path.write_text("HIERARCHY\nROOT Root\n{\nOFFSET 0 0 0\n}\nMOTION\nFrames: 1\nFrame Time: 0.033\n0 0 0\n")
    return path


def test_add_creates_motion_001(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")

    name = lib.add(src)
    assert name == "motion_001"
    assert (tmp_path / "lib" / "motion_001.bvh").is_file()
    assert (fake_ad_repo / "examples" / "bvh" / "motion_001.bvh").is_file()
    assert (fake_ad_repo / "examples" / "config" / "motion" / "motion_001.yaml").is_file()
    assert (fake_ad_repo / "examples" / "config" / "retarget" / "motion_001.yaml").is_file()


def test_motion_yaml_has_correct_filepath(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src)

    motion_yaml = (fake_ad_repo / "examples" / "config" / "motion" / "motion_001.yaml").read_text()
    assert "filepath: examples/bvh/motion_001.bvh" in motion_yaml
    assert "scale:" in motion_yaml
    assert "groundplane_joint:" in motion_yaml


def test_add_increments_counter(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src1 = make_dummy_bvh(tmp_path / "a.bvh")
    src2 = make_dummy_bvh(tmp_path / "b.bvh")
    assert lib.add(src1) == "motion_001"
    assert lib.add(src2) == "motion_002"


def test_list_returns_added_names(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.add(src)
    assert lib.list() == ["motion_001", "motion_002"]


def test_set_active_and_active(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.set_active("motion_001")
    assert lib.active() == "motion_001"


def test_active_none_when_empty(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    assert lib.active() is None


def test_get_by_index(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib.add(src)
    lib.add(src)
    assert lib.get_by_index(1) == "motion_001"
    assert lib.get_by_index(2) == "motion_002"
    assert lib.get_by_index(99) is None


def test_persistence_across_instances(tmp_path: Path, fake_ad_repo: Path):
    """Counter and listing survive process restart (re-scan disk)."""
    lib1 = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "a.bvh")
    lib1.add(src)

    lib2 = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    assert lib2.list() == ["motion_001"]
    assert lib2.add(src) == "motion_002"


def test_add_with_custom_name(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    name = lib.add(src, name="dab")
    assert name == "dab"
    assert (tmp_path / "lib" / "dab.bvh").is_file()
    assert (fake_ad_repo / "examples" / "bvh" / "dab.bvh").is_file()
    assert (fake_ad_repo / "examples" / "config" / "motion" / "dab.yaml").is_file()


def test_add_custom_name_collision_gets_suffix(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    assert lib.add(src, name="dab") == "dab"
    assert lib.add(src, name="dab") == "dab_2"
    assert lib.add(src, name="dab") == "dab_3"


def test_add_sanitizes_unsafe_chars(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    name = lib.add(src, name="my dance/01")
    # spaces and slashes replaced with _
    assert "/" not in name
    assert " " not in name


def test_list_includes_custom_names(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src)               # motion_001
    lib.add(src, name="dab")
    names = lib.list()
    assert set(names) == {"motion_001", "dab"}


def test_add_empty_name_falls_back_to_auto(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    name = lib.add(src, name="   ")  # whitespace only
    assert name == "motion_001"  # falls back to auto numbering


def test_add_preset_rokoko_writes_thigh_template(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src, name="r1", preset="rokoko")
    motion_yaml = (
        fake_ad_repo / "examples" / "config" / "motion" / "r1.yaml"
    ).read_text()
    assert "LeftThigh" in motion_yaml
    assert "RightThigh" in motion_yaml
    assert "up: +y" in motion_yaml


def test_add_preset_fair1_writes_upleg_template(tmp_path: Path, fake_ad_repo: Path):
    # Seed fair1_ppf.yaml so retarget clone has a source
    (fake_ad_repo / "examples" / "config" / "retarget" / "fair1_ppf.yaml").write_text(
        "char_starting_location: [0,0,-0.5]\n"
    )
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src, name="f1", preset="fair1")
    motion_yaml = (
        fake_ad_repo / "examples" / "config" / "motion" / "f1.yaml"
    ).read_text()
    assert "LeftUpLeg" in motion_yaml
    assert "RightUpLeg" in motion_yaml
    assert "up: +z" in motion_yaml


def test_add_preset_fair1_clones_fair1_ppf_retarget(tmp_path: Path, fake_ad_repo: Path):
    (fake_ad_repo / "examples" / "config" / "retarget" / "fair1_ppf.yaml").write_text(
        "fair1: marker\n"
    )
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src, name="f2", preset="fair1")
    retarget_yaml = (
        fake_ad_repo / "examples" / "config" / "retarget" / "f2.yaml"
    ).read_text()
    assert "fair1: marker" in retarget_yaml


def test_add_unknown_preset_falls_back_to_rokoko(tmp_path: Path, fake_ad_repo: Path):
    lib = MotionLibrary(library_dir=tmp_path / "lib", ad_repo_path=fake_ad_repo)
    src = make_dummy_bvh(tmp_path / "src.bvh")
    lib.add(src, name="x1", preset="bogus")
    motion_yaml = (
        fake_ad_repo / "examples" / "config" / "motion" / "x1.yaml"
    ).read_text()
    assert "LeftThigh" in motion_yaml  # rokoko fallback
