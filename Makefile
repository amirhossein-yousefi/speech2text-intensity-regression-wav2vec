.PHONY: format train-whisper train-wav2vec2 evaluate app

format:
	python -m black src

train-whisper:
	python -m src.speech_mtl.training.train_whisper --num_train_epochs 1

train-wav2vec2:
	python -m src.speech_mtl.training.train_wav2vec2 --num_train_epochs 1

evaluate:
	python -m src.speech_mtl.eval.evaluate

app:
	python app/gradio_app.py
