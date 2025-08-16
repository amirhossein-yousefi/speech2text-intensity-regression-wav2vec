# sagemaker/deploy_custom.py
# Upload model.tar.gz to S3 and deploy HuggingFaceModel endpoint (serverless or real-time).

import argparse
import os
import time
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def upload_to_s3(local_path, bucket, prefix):
    s3 = boto3.client("s3")
    key = f"{prefix.rstrip('/')}/{os.path.basename(local_path)}"
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint-name", default=f"speech-mtl-{int(time.time())}")
    ap.add_argument("--model-tar", required=True, help="Path to model.tar.gz")
    ap.add_argument("--s3-bucket", required=True)
    ap.add_argument("--s3-prefix", default="models/speech-mtl")
    ap.add_argument("--serverless", action="store_true", help="Use Serverless Inference")
    ap.add_argument("--memory-size", type=int, default=4096, help="Serverless memory (MB)")
    ap.add_argument("--max-concurrency", type=int, default=5)
    ap.add_argument("--instance-type", default="ml.m5.xlarge", help="If not serverless")
    ap.add_argument("--model-kind", default="wav2vec2", choices=["wav2vec2", "whisper"])
    ap.add_argument("--language", default="en")
    # DLC versions (defaults are conservative & widely available)
    ap.add_argument("--transformers-version", default="4.26")
    ap.add_argument("--pytorch-version", default="1.13")
    ap.add_argument("--py-version", default="py39")
    args = ap.parse_args()

    boto3.setup_default_session(region_name=args.region)
    sm_sess = sagemaker.Session()
    print("Uploading model...")
    model_s3 = upload_to_s3(args.model_tar, args.s3_bucket, args.s3_prefix)
    print(f"Uploaded: {model_s3}")

    env = {
        "MODEL_KIND": args.model_kind,
        "LANGUAGE": args.language,
        # You can pass any other env your inference code needs here.
    }

    # Build model object. You can also supply entry_point/source_dir instead of bundling code/
    # in the model.tar.gz. We keep code inside model.tar.gz here to mirror blog guidance.
    # (Either pattern is supported by HF on SageMaker.)
    model = HuggingFaceModel(
        model_data=model_s3,
        role=args.role_arn,
        env=env,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        py_version=args.py_version,
    )

    if args.serverless:
        predictor = model.deploy(
            endpoint_name=args.endpoint_name,
            serverless_config={"memory_size_in_mb": args.memory_size, "max_concurrency": args.max_concurrency},
        )
    else:
        predictor = model.deploy(
            endpoint_name=args.endpoint_name,
            initial_instance_count=1,
            instance_type=args.instance_type,
        )

    print("Deployed endpoint:", predictor.endpoint_name)
    print("Invoke example:")
    print(
        "python sagemaker/client/invoke.py "
        f"--region {args.region} --endpoint {predictor.endpoint_name} --wav path/to/sample.wav"
    )

if __name__ == "__main__":
    main()
