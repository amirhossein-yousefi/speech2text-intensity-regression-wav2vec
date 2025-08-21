import os
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace

# --- Resolve role ---
def resolve_role():
    try:
        return sagemaker.get_execution_role()  # works inside SageMaker
    except Exception:
        role_name = os.getenv("SAGEMAKER_EXECUTION_ROLE_NAME", "SageMakerExecutionRole")
        return boto3.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]

if __name__ == '__main__':
    sess   = sagemaker.Session()
    region = sess.boto_region_name
    role   = resolve_role()

    # --- Use latest HF PyTorch Training DLC (GPU) ---
    # See: "Available DLCs on AWS" (kept up-to-date by HF)
    # https://huggingface.co/docs/sagemaker/en/dlcs/available
    account = "763104351884"
    image_tag = "2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04"
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:{image_tag}"

    # second option:
    # estimator = HuggingFace(
    #   transformers_version="4.49.0", pytorch_version="2.5.1", py_version="py311", ...)

    # --- Training hyperparameters: map to sagemaker/training/code/train.py arguments ---
    hyperparameters = {
        # choose "wav2vec2" or "whisper"
        "model": "wav2vec2",
        "model_name": "facebook/wav2vec2-base-960h",
        "dataset": "librispeech_asr",
        "train_split": "train.clean.100",
        "eval_split": "validation.clean",
        "text_column": "text",
        "language": "en",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-5,
        "lambda_intensity": 1.0,
        "save_total_limit": 1,
        # save to /opt/ml/model so SageMaker uploads it as model.tar.gz
        "output_dir": "/opt/ml/model",
        "fp16": "true",
        "report_to": "none",
    }

    # Spot + checkpointing (recommended for cost savings) ---
    checkpoint_s3 = f"s3://{sess.default_bucket()}/checkpoints/speech_mtl/"
    use_spot = True
    max_run_seconds = 6 * 60 * 60      # 6h
    max_wait_seconds = max_run_seconds # must be >= max_run when use_spot_instances=True

    # Parse CloudWatch metrics from HF logs ---
    metric_definitions = [
        {"Name": "train_runtime",         "Regex": r"train_runtime.*=\D*(.*?)$"},
        {"Name": "eval_loss",             "Regex": r"eval_loss.*=\D*(.*?)$"},
        {"Name": "eval_wer",              "Regex": r"eval_wer.*=\D*(.*?)$"},
        {"Name": "eval_intensity_mae",    "Regex": r"eval_intensity_mae.*=\D*(.*?)$"},
        {"Name": "eval_intensity_rmse",   "Regex": r"eval_intensity_rmse.*=\D*(.*?)$"},
    ]


    estimator = HuggingFace(
        role=role,
        image_uri=image_uri,
        instance_type=os.getenv("SM_TRAIN_INSTANCE_TYPE", "ml.g5.2xlarge"),
        instance_count=int(os.getenv("SM_TRAIN_INSTANCE_COUNT", "1")),
        source_dir=".",
        entry_point="sagemaker/training/train_entry.py",
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        # storage & cost controls
        volume_size=int(os.getenv("SM_TRAIN_VOLUME_SIZE_GB", "200")),
        max_run=max_run_seconds,
        use_spot_instances=use_spot,
        max_wait=max_wait_seconds if use_spot else None,
        checkpoint_s3_uri=checkpoint_s3 if use_spot else None,

    )


    job_name = os.getenv("SM_TRAIN_JOB_NAME", sagemaker.utils.name_from_base("speech-mtl-train"))
    print(f"Launching training job: {job_name}")
    estimator.fit(job_name=job_name, wait=True)

    print("Training complete.")
    print(f"Model artifacts: {estimator.model_data}")
