from __future__ import annotations
import argparse, os
from dataclasses import dataclass
import numpy as np
import torch
from transformers import (
    WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, AutoConfig
)
from datasets import DatasetDict
from ..data.datasets import load_asr_dataset_with_intensity
from ..data.collators import DataCollatorSpeechSeq2SeqWithPadding
from ..models.multitask_whisper import WhisperForASRAndIntensity

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', type=str, default='openai/whisper-small')
    ap.add_argument('--language', type=str, default='en')
    ap.add_argument('--dataset', type=str, default='librispeech_asr')
    ap.add_argument('--train_split', type=str, default='train.clean.100')
    ap.add_argument('--eval_split', type=str, default='validation.clean')
    ap.add_argument('--text_column', type=str, default='text')
    ap.add_argument('--output_dir', type=str, default='outputs/whisper_mtl')
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--learning_rate', type=float, default=1.5e-4)
    ap.add_argument('--per_device_train_batch_size', type=int, default=8)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=8)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=2)
    ap.add_argument('--warmup_steps', type=int, default=500)
    ap.add_argument('--fp16', action='store_true', default=False)
    ap.add_argument('--lambda_intensity', type=float, default=1.0)
    ap.add_argument('--generation_max_new_tokens', type=int, default=128)
    return ap.parse_args()

def main():
    args = parse_args()
    ds_train, ds_eval, _ = load_asr_dataset_with_intensity(
        dataset=args.dataset,
        language=args.language if args.dataset != 'librispeech_asr' else None,
        train_split=args.train_split,
        eval_split=args.eval_split,
        test_split=None,
        text_column=args.text_column,
    )

    # Processor (feature extractor + tokenizer)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

    # Preprocess: log-mel features + labels
    def prepare_example(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Labels
        with tokenizer.as_target_tokenizer():
            batch["labels"] = tokenizer(batch[args.text_column]).input_ids
        return batch

    ds_train = ds_train.map(prepare_example, remove_columns=[c for c in ds_train.column_names if c not in ["input_features","labels","intensity_norm"]], desc="Preparing Whisper features (train)")
    ds_eval  = ds_eval.map(prepare_example, remove_columns=[c for c in ds_eval.column_names  if c not in ["input_features","labels","intensity_norm"]],  desc="Preparing Whisper features (eval)")

    # Model
    config = AutoConfig.from_pretrained(args.model_name)
    model = WhisperForASRAndIntensity.from_pretrained(args.model_name, config=config)
    model.generation_config = GenerationConfig.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None  # we'll pass bos via collator

    # Data collator
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    # Metrics (decode & WER + intensity RMSE/MAE)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # preds: logits, we generate separately for text; here we only compute intensity metrics using labels trick
        # In Seq2SeqTrainer, eval_pred[0] are logits, not used for text metrics here.
        return {}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        predict_with_generate=True,
        generation_max_length=args.generation_max_new_tokens,
        fp16=args.fp16,
        report_to=["none"],
    )

    class MTTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Pass intensity weight
            outputs = model(**inputs, lambda_intensity=float({{lambda_intensity}}))
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    # Hack lambda into f-string safely:
    MTTrainer.compute_loss.__defaults__ = None  # ensure signature stable
    # Recreate class to inject args.lambda_intensity cleanly
    class MTTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs, lambda_intensity=float(args.lambda_intensity))
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = MTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        tokenizer=processor.feature_extractor,  # for logging shapes
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training done. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
