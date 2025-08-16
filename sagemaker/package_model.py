# Create a SageMaker-ready model.tar.gz containing your HF model weights
import argparse
import os
import tarfile
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Path to fine-tuned HF model dir (e.g., outputs/wav2vec2_base_mtl)")
    ap.add_argument("--out", default="model.tar.gz", help="Output tarball path")
    ap.add_argument("--include-src", action="store_true", help="Include src/speech_mtl in code/ for custom classes")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    assert model_dir.exists(), f"Model dir not found: {model_dir}"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Copy model artifacts (config.json, pytorch_model.bin, tokenizer, etc.)
        dst_model = tmp
        for p in model_dir.glob("*"):
            if p.is_dir():
                shutil.copytree(p, dst_model / p.name)
            else:
                shutil.copy2(p, dst_model / p.name)

        # Add code/ with inference.py (+ optional requirements.txt and src code)
        code_dir = tmp / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # inference.py + requirements.txt from sagemaker/code/
        sm_code = HERE / "code"
        for name in ["inference.py", "requirements.txt"]:
            src = sm_code / name
            if src.exists():
                shutil.copy2(src, code_dir / name)

        # Optionally vendor your library code so custom classes import cleanly
        if args.include_src:
            repo_root = HERE.parent
            src_pkg = repo_root / "src" / "speech_mtl"
            if src_pkg.exists():
                dst_pkg_root = code_dir / "speech_mtl"
                shutil.copytree(src_pkg, dst_pkg_root)

        # Tar it up
        out = Path(args.out).resolve()
        if out.exists():
            out.unlink()
        with tarfile.open(out, "w:gz") as tar:
            tar.add(tmp, arcname=".")
        print(f"Created {out}")

if __name__ == "__main__":
    main()
