import argparse
import base64
import boto3
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--wav", required=True)
    args = ap.parse_args()

    boto3.setup_default_session(region_name=args.region)
    runtime = boto3.client("sagemaker-runtime")

    wav_path = Path(args.wav)
    data = {
        "audio_base64": base64.b64encode(wav_path.read_bytes()).decode("utf-8")
    }

    resp = runtime.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(data).encode("utf-8"),
    )
    body = resp["Body"].read()
    print(json.loads(body))

if __name__ == "__main__":
    main()
