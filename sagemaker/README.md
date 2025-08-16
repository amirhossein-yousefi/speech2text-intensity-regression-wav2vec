# SageMaker deployment (serverless or real-time)

This adds a production-style deploy for the multitask **ASR + intensity** model using the Hugging Face SageMaker DLCs with a custom `inference.py`.

## 0) Prereqs

- AWS account with SageMaker access
- An **execution role** with permissions to create/endpoints (pass its ARN below)
- An S3 bucket for model artifacts (e.g., `s3://my-ml-models`)
- A trained checkpoint at `outputs/<your_model_dir>` (see repo README).  <!-- training scripts already exist in this repo -->  

## 1) Package your model

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # local deps, optional
pip install sagemaker boto3

# package the wav2vec2 (or whisper) checkpoint into model.tar.gz with custom code/
python sagemaker/package_model.py \
  --model-dir outputs/wav2vec2_base_mtl \
  --out model.tar.gz \
  --include-src
  ```
## 2) Deploy 
```
python sagemaker/deploy_custom.py \
  --region us-east-1 \
  --role-arn arn:aws:iam::<account-id>:role/<SageMakerExecutionRole> \
  --endpoint-name speech-mtl-demo \
  --model-tar model.tar.gz \
  --s3-bucket <your-bucket> \
  --s3-prefix models/speech-mtl \
  --serverless --memory-size 4096 --max-concurrency 5 \
  --model-kind wav2vec2 --language en
```
## 2) Invoke
```
python sagemaker/client/invoke.py \
  --region us-east-1 \
  --endpoint speech-mtl-demo \
  --wav ./sample.wav
```
## Returns 
```
{"text": "example transcript", "intensity_norm": 0.42, "intensity_dbfs": -25.2}
```