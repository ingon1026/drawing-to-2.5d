# CLAUDE.md — stickerbook

## Project context
Stage 1 Python PC PoC for the "2.5D 증강현실 (AR Stickerbook)" sub-project of AR_book.

Goal: a user points a webcam at a child's drawing on a sketchbook/notebook page, clicks the drawing, and the clicked drawing is extracted as a 2.5D sticker that stays world-anchored (via homography tracking) on the paper location. Later stages add pop-up (standing) sticker rendering and tablet deployment.

This PC Python project covers the full pipeline — detection, segmentation, tracking, rendering, asset export. Tablet (Galaxy / ARCore) porting is a future follow-up using the same Python stack (ONNX + Android native), not a separate engine.

See `docs/DESIGN.md` for the full design spec.

## References
- Main rigging/animation base: https://github.com/facebookresearch/AnimatedDrawings (MIT)
- 2.5D sticker AR inspiration: https://github.com/tatsuya-ogawa/RakugakiAR (Swift/ARKit)
- Sibling PoC code (model reuse): `../LivingDrawing/`

## Module boundaries
Each module under `capture/`, `detect/`, `extract/`, `track/`, `render/`, `export/` has a single narrow responsibility and a documented interface. Cross-module communication goes through the dataclasses defined in `render/sticker.py` and `track/world_anchor.py`.

Do not merge modules without updating `docs/DESIGN.md` first.

## Implementation rules for stickerbook
- Stage 1 only — keep this folder as PC Python PoC. Mobile/ARCore code lives in a sibling folder if/when added, never here.
- SAM inference must run on a background thread; the main OpenCV loop never blocks on SAM.
- `WorldAnchor` is a Protocol. Only one concrete implementation in Stage 1 (`HomographyAnchor`). A second implementation (`ArUcoAnchor`) is an allowed fallback but must land behind the same interface.
- Exported assets must round-trip: loading `assets/captures/<timestamp>/` in the AnimatedDrawings local demo must not error.
- Any new dependency goes into `requirements.txt`. No ad-hoc `pip install`.

## Milestones
Work proceeds M1 → M7 in `docs/DESIGN.md`. Do not ship M2 before M1 is demoable.

## Avoid
- Mocking the webcam or SAM inside tests that are supposed to verify live behavior
- Abstract base classes for single-use interfaces (use `typing.Protocol`)
- Long docstrings; one-line comments only when the "why" is non-obvious
- Feature creep beyond the Stage 1 scope in `docs/DESIGN.md`
