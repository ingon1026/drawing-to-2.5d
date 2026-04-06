"""
drawing-2p5d — CLI pipeline entry point

Usage:
    python3 pipeline.py --input samples/test.jpg --x 0.5 --y 0.4
    python3 pipeline.py --input samples/test.jpg --x 0.5 --y 0.4 --output results/ --debug
"""

import argparse
import os
import sys

import config
import normalize
import segment
import postprocess
import depth
import export


def main():
    parser = argparse.ArgumentParser(description="2.5D drawing asset pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--x", type=float, required=True, help="Normalized x (0~1, left to right)")
    parser.add_argument("--y", type=float, required=True, help="Normalized y (0~1, top to bottom)")
    parser.add_argument("--output", "-o", default=config.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--debug", "-d", action="store_true", help="Export debug overlay")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--normal-strength", type=float, default=config.NORMAL_STRENGTH,
                        help="Normal map strength (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    # 1. Download segmentation model if needed
    print("[1/7] Checking segmentation model ...")
    segment.download_model_if_needed()

    # 2. Normalize input
    print("[2/7] Normalizing input ...")
    image = normalize.normalize_input(args.input)
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # 3. Segment
    print(f"[3/7] Segmenting at point ({args.x:.3f}, {args.y:.3f}) ...")
    segmenter = segment.load_segmenter()
    raw_mask = segment.segment_at_point(segmenter, image, args.x, args.y, args.threshold)
    fg_ratio = raw_mask.sum() / (raw_mask.size * 255) * 100
    print(f"  Raw mask foreground: {fg_ratio:.1f}%")

    # 4. Postprocess mask
    print("[4/7] Postprocessing mask ...")
    mask = postprocess.clean_mask(raw_mask)
    fg_ratio = mask.sum() / (mask.size * 255) * 100
    print(f"  Clean mask foreground: {fg_ratio:.1f}%")

    # 5. Depth estimation
    print("[5/7] Estimating depth map ...")
    depth_map = depth.estimate_depth(image, mask)
    print(f"  Depth range: [{depth_map[mask > 0].min():.3f}, {depth_map[mask > 0].max():.3f}]")

    # 6. Normal map from depth
    print("[6/7] Generating normal map ...")
    normal_map = depth.depth_to_normal(depth_map, strength=args.normal_strength)
    # Zero out normals outside mask
    normal_map[mask == 0] = (128, 128, 255)  # neutral flat normal outside mask

    # 7. Export all assets
    print("[7/7] Exporting 2.5D assets ...")
    mask_path = export.export_mask(mask, args.output)
    obj_path = export.export_object(image, mask, args.output)
    depth_path = export.export_depth(depth_map, args.output)
    normal_path = export.export_normal(normal_map, args.output)
    print(f"  {mask_path}")
    print(f"  {obj_path}")
    print(f"  {depth_path}")
    print(f"  {normal_path}")

    if args.debug:
        dbg_path = export.export_debug_overlay(image, mask, args.output)
        print(f"  {dbg_path}")

    print("Done. 2.5D assets ready.")


if __name__ == "__main__":
    main()
