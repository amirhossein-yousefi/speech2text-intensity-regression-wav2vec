from __future__ import annotations
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class WhisperForASRAndIntensity(WhisperForConditionalGeneration):
    """Whisper with an additional regression head for intensity.
    Uses mean-pooled encoder states -> MLP -> scalar.
    """
    def __init__(self, config):
        super().__init__(config)
        hidden = config.d_model
        self.intensity_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mse = nn.MSELoss()

    def forward(
        self,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        intensity_value: Optional[torch.FloatTensor] = None,
        lambda_intensity: float = 1.0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        # Run base forward to get ASR loss/logits and encoder states
        outputs = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Encoder last hidden state: (B, T, H)
        if outputs.encoder_last_hidden_state is not None:
            enc = outputs.encoder_last_hidden_state
            pooled = enc.mean(dim=1)  # (B, H), Whisper encoder has no padding mask in features
            intensity_pred = self.intensity_head(pooled).squeeze(-1)  # (B,)
        else:
            intensity_pred = None

        asr_loss = outputs.loss if getattr(outputs, "loss", None) is not None else None
        intensity_loss = None
        if (intensity_pred is not None) and (intensity_value is not None):
            intensity_loss = self.mse(intensity_pred, intensity_value)

        loss = None
        if (asr_loss is not None) and (intensity_loss is not None):
            loss = asr_loss + lambda_intensity * intensity_loss
        elif asr_loss is not None:
            loss = asr_loss
        elif intensity_loss is not None:
            loss = lambda_intensity * intensity_loss

        if not return_dict:
            out = list(outputs)
            if intensity_pred is not None:
                out.append(intensity_pred)
            if loss is not None:
                out[0] = loss  # first element is loss
            return tuple(out)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
