# SageMaker Training

This folder contains the scripts to **launch a SageMaker training job** for the
multi-task speech models in this repo (ASR + intensity regression).

## Prereqs

- AWS account + IAM role with SageMaker permissions
- `pip install sagemaker boto3`
- Optional: configure [Managed Spot Training + checkpointing]

## Quickstart (Wav2Vec2)

From the repo root:

```bash
python sagemaker/train.py \
  # override defaults via env vars if needed:
  # SM_TRAIN_INSTANCE_TYPE=ml.g5.2xlarge \
  # SM_TRAIN_VOLUME_SIZE_GB=200 \
  # SM_TRAIN_JOB_NAME=my-wav2vec2-mtl \
  # SAGEMAKER_EXECUTION_ROLE_NAME=SageMakerExecutionRole \
  # hyperparameters are defined inside sagemaker/train.py under `hyperparameters`
