# Speech Multitask End-to-End (ASR + Intensity Regression)

This project trains and serves **two** end-to-end speech models that jointly perform:
- **Speech-to-Text (ASR)** and
- **Voice intensity regression** (in dBFS normalized to [0,1]) from the same audio.


## Project Structure
```
speech_mtl_end2end/
├── README.md
├── requirements.txt
├── setup.cfg
├── Makefile
├── .gitignore
├── LICENSE
├── configs/
│   ├── whisper_base.yaml
│   └── wav2vec2_base.yaml
├── app/
│   └── gradio_app.py
├── src/
│   └── speech_mtl/
│       ├── __init__.py
│       ├── utils/
│       │   ├── audio.py
│       │   └── metrics.py
│       ├── data/
│       │   ├── datasets.py
│       │   └── collators.py
│       ├── models/
│       │   ├── multitask_whisper.py
│       │   └── multitask_wav2vec2.py
│       ├── training/
│       │   ├── train_whisper.py
│       │   └── train_wav2vec2.py
│       ├── eval/
│       │   └── evaluate.py
│       └── inference/
│           └── predict.py
└── tests/
    └── test_audio_utils.py
```

## Approaches
1. **Whisper + Regression head** (Seq2Seq): `src/speech_mtl/models/multitask_whisper.py`
2. **Wav2Vec2-CTC + Regression head**: `src/speech_mtl/models/multitask_wav2vec2.py`

Both are trained using Hugging Face `transformers` and `datasets`.

## Dataset
Default: [Librispeech ASR](https://huggingface.co/datasets/librispeech_asr) (`train.clean.100` / `validation.clean` / `test.clean`).  
We compute **intensity labels** from the audio as **RMS dBFS** (bounded to [-60, 0]) and also provide a normalized target in `[0,1]`:
```
norm_intensity = (dbfs + 60) / 60
```
Optionally, you can switch to **LUFS** (requires `pyloudnorm`).

## Quickstart

### 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(Optional) For GPU training, also install a CUDA build of PyTorch per https://pytorch.org/get-started/locally/

### 1) Train Whisper (ASR + intensity)
```bash
python -m src.speech_mtl.training.train_whisper \
  --model_name openai/whisper-small \
  --language en \
  --dataset librispeech_asr \
  --train_split train.clean.100 \
  --eval_split validation.clean \
  --text_column text \
  --num_train_epochs 1 \
  --output_dir outputs/whisper_small_mtl
```

### 2) Train Wav2Vec2-CTC (ASR + intensity)
```bash
python -m src.speech_mtl.training.train_wav2vec2 
   --model_name facebook/wav2vec2-base-960h 
   --dataset librispeech_asr 
   --train_split train.clean.100 
   --eval_split validation.clean 
   --text_column text 
   --max_train_samples 1000 
   --max_eval_samples 150 
   --num_train_epochs 1 
   --output_dir outputs\wav2vec2_small
```

### 3) Evaluate
```bash
python -m src.speech_mtl.eval.evaluate \
  --whisper_model_dir outputs/whisper_small_mtl \
  --wav2vec2_model_dir outputs/wav2vec2_base_mtl \
  --dataset librispeech_asr --split test.clean --text_column text
```

### 4) Inference (CLI)
```bash
python -m src.speech_mtl.inference.predict \
  --model whisper \
  --checkpoint outputs/whisper_small_mtl \
  --audio path/to/audio.wav
```

### 5) Gradio Demo
```bash
python app/gradio_app.py --model whisper --checkpoint outputs/whisper_small_mtl
# or
python app/gradio_app.py --model wav2vec2 --checkpoint outputs/wav2vec2_base_mtl
```



## Notes
- **Intensity loss weight** is controlled by `--lambda_intensity`. Set `0.0` to disable.
- For **Whisper**, we use a mean-pooled encoder representation for the regression head.
- For **Wav2Vec2-CTC**, we use attention-masked mean pooling for the regression head.
- You can swap `librispeech_asr` with `mozilla-foundation/common_voice_13_0` by passing `--dataset` and `--language` accordingly.

## License
MIT
