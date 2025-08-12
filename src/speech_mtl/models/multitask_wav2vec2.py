from __future__ import annotations
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
try:
    from transformers.modeling_outputs import CTCOutput  # older versions
except ImportError:
    from dataclasses import dataclass
    from typing import Optional, Tuple
    import torch
    from transformers.modeling_outputs import ModelOutput

    @dataclass
    class CTCOutput(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
class Wav2Vec2ForCTCAndIntensity(Wav2Vec2ForCTC):
    """Wav2Vec2-CTC with an additional regression head for intensity.
    Pools the last hidden state with attention mask then MLP -> scalar.
    """
    def __init__(self, config):
        super().__init__(config)
        hidden = config.hidden_size
        self.intensity_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.mse = nn.MSELoss()

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        intensity_value: Optional[torch.FloatTensor] = None,
        lambda_intensity: float = 1.0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Tuple, CTCOutput]:
        outputs = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        # Use last hidden state for regression: (B, T, H)
        hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        intensity_pred = None
        if hidden is not None:
            if attention_mask is not None:
                # Masked mean pooling over time
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                pooled = hidden.mean(dim=1)
            intensity_pred = self.intensity_head(pooled).squeeze(-1)

        ctc_loss = outputs.loss if getattr(outputs, "loss", None) is not None else None
        intensity_loss = None
        if (intensity_pred is not None) and (intensity_value is not None):
            intensity_loss = self.mse(intensity_pred, intensity_value)

        loss = None
        if (ctc_loss is not None) and (intensity_loss is not None):
            loss = ctc_loss + lambda_intensity * intensity_loss
        elif ctc_loss is not None:
            loss = ctc_loss
        elif intensity_loss is not None:
            loss = lambda_intensity * intensity_loss

        if not return_dict:
            out = list(outputs)
            if intensity_pred is not None:
                out.append(intensity_pred)
            if loss is not None:
                out[0] = loss
            return tuple(out)

        return CTCOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
