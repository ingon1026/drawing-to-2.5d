"""
drawing-2p5d — MiDaS DPT-hybrid → TFLite 변환

변환 경로: PyTorch → ONNX → TFLite
출력: models/midas_depth.tflite

Usage:
    python3 export_tflite.py
"""

import os
import numpy as np
import torch
import onnx

OUTPUT_DIR = "models"
ONNX_PATH = os.path.join(OUTPUT_DIR, "midas_depth.onnx")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "midas_depth.tflite")
INPUT_SIZE = 384  # DPT-hybrid default input size


def step1_pytorch_to_onnx():
    """MiDaS PyTorch → ONNX 변환"""
    print("[1/3] PyTorch → ONNX ...")

    from transformers import DPTForDepthEstimation

    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes=None,  # 고정 크기 (모바일 최적화)
        opset_version=17,
    )

    # Verify
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  ONNX saved: {ONNX_PATH} ({size_mb:.1f} MB)")


def step2_onnx_to_tflite():
    """ONNX → TFLite 변환 (float32)"""
    print("[2/3] ONNX → TFLite ...")

    import subprocess
    result = subprocess.run(
        [
            "onnx2tf",
            "-i", ONNX_PATH,
            "-o", os.path.join(OUTPUT_DIR, "tflite_tmp"),
            "-oiqt",  # INT8 quantized model도 생성
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  onnx2tf error:\n{result.stderr[:500]}")
        # Fallback: tf.lite.TFLiteConverter 직접 사용
        print("  Trying fallback conversion via TF SavedModel ...")
        fallback_onnx_to_tflite()
        return

    # Find the generated tflite file
    tmp_dir = os.path.join(OUTPUT_DIR, "tflite_tmp")
    for f in os.listdir(tmp_dir):
        if f.endswith(".tflite") and "float32" in f:
            src = os.path.join(tmp_dir, f)
            os.rename(src, TFLITE_PATH)
            break
    else:
        # Just take the first .tflite
        for f in os.listdir(tmp_dir):
            if f.endswith(".tflite"):
                src = os.path.join(tmp_dir, f)
                os.rename(src, TFLITE_PATH)
                break

    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"  TFLite saved: {TFLITE_PATH} ({size_mb:.1f} MB)")


def fallback_onnx_to_tflite():
    """Fallback: ONNX → TF SavedModel → TFLite"""
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(ONNX_PATH)
    tf_rep = prepare(onnx_model)

    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model_tmp")
    tf_rep.export_graph(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"  TFLite saved: {TFLITE_PATH} ({size_mb:.1f} MB)")


def step3_verify():
    """TFLite 모델 검증 — 입출력 shape 확인 + 더미 추론"""
    print("[3/3] TFLite 검증 ...")

    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  {input_details[0]['name']} shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['name']} shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    # Dummy inference
    dummy = np.random.rand(*input_details[0]['shape']).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Dummy inference OK, output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"\n  Done! Model ready: {TFLITE_PATH}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    step1_pytorch_to_onnx()
    step2_onnx_to_tflite()
    step3_verify()
