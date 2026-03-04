# Speech Recognition

A keyword/speech command recognition system built with PyTorch, trained on the [Google Speech Commands v0.02](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) dataset. The project includes model training notebooks and a Flask web application for real-time inference.

---

## Project Structure

```
Speech-Recognition/
├── app.py                  # Flask web server for real-time inference
├── precompute_mfcc.py      # MFCC feature pre-computation script
├── main.ipynb              # CNN-LSTM model training notebook
├── cnn.ipynb               # CNN-only model training notebook
├── gru.ipynb               # GRU model training notebook
├── best_model.pth          # Best CNN-LSTM model weights
├── best_model_cnn.pth      # Best CNN model weights
├── best_model_gru.pth      # Best GRU model weights
├── templates/
│   └── index.html          # Web UI for audio recording and prediction
└── data/                   # Dataset directory (created on download)
    └── SpeechCommands/
        └── speech_commands_v0.02/
            └── mfcc_cache/ # Pre-computed MFCC features
```

---

## Models

All models take 40-dimensional MFCC features over 100 time steps as input.

| Model | Architecture | Weights |
|---|---|---|
| CNN-LSTM | 2× Conv1D → Bidirectional LSTM (2 layers, 128 hidden) → FC | `best_model.pth` |
| CNN | Convolutional neural network | `best_model_cnn.pth` |
| GRU | Gated Recurrent Unit network | `best_model_gru.pth` |

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, CPU is supported)

### Install Dependencies

```bash
pip install torch torchaudio librosa soundfile numpy flask tqdm
```

### Download the Dataset

Run the first cell in `main.ipynb`, or execute:

```python
import torchaudio
torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)
```

### Pre-compute MFCC Features

This step is required before training. It processes all audio files and caches MFCC features to disk as a memory-mapped file for fast loading during training.

```bash
python precompute_mfcc.py
```

This creates `./data/SpeechCommands/speech_commands_v0.02/mfcc_cache/` containing:
- `mfccs.dat` — memory-mapped MFCC array (~GB scale)
- `labels.npy`, `meta.npy`, `files.npy`, `classes.npy`

---

## Training

Open and run the appropriate notebook:

| Notebook | Model |
|---|---|
| `main.ipynb` | CNN-LSTM (recommended) |
| `cnn.ipynb` | CNN |
| `gru.ipynb` | GRU |

Trained weights are saved as `best_model.pth`, `best_model_cnn.pth`, and `best_model_gru.pth` respectively.

---

## Running the Web App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

The web interface allows you to:
- Record audio directly from your microphone
- Upload a `.wav` audio file
- View the predicted speech command and confidence score

### API

**POST** `/predict`

Upload an audio file to get a prediction.

```bash
curl -X POST http://localhost:5000/predict \
  -F "audio=@your_audio.wav"
```

**Response:**
```json
{
  "prediction": "yes",
  "confidence": 97.3
}
```

---

## Feature Extraction

- **Sample rate:** 16 kHz
- **Features:** 40 MFCC coefficients
- **Sequence length:** 100 frames (padded or truncated)
- **Input shape to model:** `(batch, 100, 40)`
