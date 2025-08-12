from __future__ import annotations
import argparse, os, numpy as np, torch
from transformers import Wav2Vec2Processor, AutoConfig, TrainingArguments, Trainer
from ..data.datasets import load_asr_dataset_with_intensity
from ..data.collators import DataCollatorCTCWithPaddingAndIntensity
from ..models.multitask_wav2vec2 import Wav2Vec2ForCTCAndIntensity
from jiwer import wer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h')
    ap.add_argument('--language', type=str, default='en')
    ap.add_argument('--dataset', type=str, default='librispeech_asr')
    ap.add_argument('--train_split', type=str, default='train.clean.100')
    ap.add_argument('--eval_split', type=str, default='validation.clean')
    ap.add_argument('--text_column', type=str, default='text')
    ap.add_argument('--output_dir', type=str, default='outputs/wav2vec2_mtl')
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--per_device_train_batch_size', type=int, default=8)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=8)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--warmup_steps', type=int, default=500)
    ap.add_argument('--fp16', action='store_true', default=False)
    ap.add_argument('--lambda_intensity', type=float, default=1.0)
    ap.add_argument('--ctc_zero_infinity', action='store_true', default=True)
    ap.add_argument('--pad_to_multiple_of', type=int, default=None)
    ap.add_argument('--cache_dir', type=str, default=None)
    # NEW: tiny dataset knobs
    ap.add_argument('--max_train_samples', type=int, default=None)
    ap.add_argument('--max_eval_samples', type=int, default=None)
    ap.add_argument('--max_test_samples', type=int, default=None)
    ap.add_argument('--num_proc', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()

    # Load data with caps applied BEFORE intensity/decoding for speed
    ds_train, ds_eval, _ = load_asr_dataset_with_intensity(
        dataset=args.dataset,
        language=args.language if args.dataset not in ('librispeech_asr',) else None,
        train_split=args.train_split,
        eval_split=args.eval_split,
        test_split=None,
        text_column=args.text_column,
        cache_dir=args.cache_dir,
        num_proc=int(args.num_proc),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    # Processor (feature extractor + tokenizer for CTC)
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    # Preprocess: input_values + labels (CTC)
    # --- replace your prepare_example with this ---
    def prepare_example(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]

        # Avoid Processor.__call__ for text; use the tokenizer directly to dodge the
        # 'return_attention_mask' duplication bug.
        text = batch[args.text_column]
        if not isinstance(text, str):
            text = str(text)
        batch["labels"] = processor.tokenizer(
            text, add_special_tokens=False
        ).input_ids
        return batch

    keep_cols = ["input_values", "labels", "intensity_norm"]
    ds_train = ds_train.map(
        prepare_example,
        remove_columns=[c for c in ds_train.column_names if c not in keep_cols],
        desc="Preparing Wav2Vec2 features (train)"
    )
    ds_eval = ds_eval.map(
        prepare_example,
        remove_columns=[c for c in ds_eval.column_names if c not in keep_cols],
        desc="Preparing Wav2Vec2 features (eval)"
    )

    config = AutoConfig.from_pretrained(args.model_name)
    config.ctc_zero_infinity = bool(args.ctc_zero_infinity)
    model = Wav2Vec2ForCTCAndIntensity.from_pretrained(args.model_name, config=config)

    collator = DataCollatorCTCWithPaddingAndIntensity(processor=processor)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, group_tokens=False)
        asr_wer = wer(label_str, pred_str)
        return {"wer": asr_wer}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        fp16=args.fp16,
        report_to=["none"],
        remove_unused_columns=False,  # keep intensity_norm for collator/model
    )

    class MTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs, lambda_intensity=float(args.lambda_intensity))
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = MTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training done. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
