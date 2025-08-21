import argparse, os, sys, subprocess, json, pathlib


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="wav2vec2",
                        choices=["wav2vec2", "whisper"],
                        help="Which training script to use from src/speech_mtl/training/*")

    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--dataset", type=str, default="librispeech_asr")
    parser.add_argument("--train_split", type=str, default="train.clean.100")
    parser.add_argument("--eval_split", type=str, default="validation.clean")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--language", type=str, default="en")  # whisper only

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--lambda_intensity", type=float, default=1.0)

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=str, default="true")
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Ensure artifacts end up in SM_MODEL_DIR so SageMaker uploads them to S3
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--report_to", type=str, default="none")

    args = parser.parse_args()

    module = {
        "wav2vec2": "src.speech_mtl.training.train_wav2vec2",
        "whisper": "src.speech_mtl.training.train_whisper",
    }[args.model]

    cmd = [
        sys.executable, "-m", module,
        "--model_name", args.model_name,
        "--dataset", args.dataset,
        "--train_split", args.train_split,
        "--eval_split", args.eval_split,
        "--text_column", args.text_column,
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--lambda_intensity", str(args.lambda_intensity),
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
        "--save_total_limit", str(args.save_total_limit),
        "--report_to", args.report_to,
    ]

    if args.model == "whisper":
        cmd += ["--language", args.language]

    # Optional flags only if provided
    if args.max_train_samples and args.max_train_samples > 0:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if args.max_eval_samples and args.max_eval_samples > 0:
        cmd += ["--max_eval_samples", str(args.max_eval_samples)]
    if args.resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", args.resume_from_checkpoint]

    if str2bool(args.fp16):
        cmd += ["--fp16", "true"]

    os.environ.setdefault("TRANSFORMERS_CACHE", "/opt/ml/input/.cache/hf")
    os.environ.setdefault("HF_HOME", "/opt/ml/input/.cache/hf")
    pathlib.Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)

    print(f"[trainer] Launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    meta = {
        "model": args.model,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "text_column": args.text_column,
        "language": args.language,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "sagemaker_train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
