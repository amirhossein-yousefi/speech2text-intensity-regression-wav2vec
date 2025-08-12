from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch

# -------- Whisper Seq2Seq collator --------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any
    decoder_start_token_id: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pull out intensity
        intensity = [f["intensity_norm"] for f in features]
        has_labels = "labels" in features[0]

        # Prepare inputs
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if has_labels:
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

        batch["intensity_value"] = torch.tensor(intensity, dtype=torch.float32)
        if self.decoder_start_token_id is not None:
            # force BOS
            batch["decoder_input_ids"] = torch.full(
                (len(features), 1), self.decoder_start_token_id, dtype=torch.long
            )
        return batch

# -------- Wav2Vec2 CTC collator --------
@dataclass
class DataCollatorCTCWithPaddingAndIntensity:
    processor: any
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        intensity = [f["intensity_norm"] for f in features]
        input_features = [{"input_values": f["input_values"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        if "labels" in features[0]:
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
            # Replace padding with -100 to ignore in loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

        batch["intensity_value"] = torch.tensor(intensity, dtype=torch.float32)
        return batch
