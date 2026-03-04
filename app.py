import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import librosa
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify, render_template

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(40, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

CLASSES = sorted([
    d for d in os.listdir("./data/SpeechCommands/speech_commands_v0.02")
    if os.path.isdir(os.path.join("./data/SpeechCommands/speech_commands_v0.02", d))
    and not d.startswith("_")
    and d != "mfcc_cache"
])
NUM_CLASSES = len(CLASSES)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_LSTM(NUM_CLASSES).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print(f"Model loaded on {device} with {NUM_CLASSES} classes")

def extract_mfcc(audio_path, max_len=100):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T  

def extract_mfcc_from_bytes(audio_bytes, sr=16000, max_len=100):
    """Extract MFCC from raw audio bytes (WAV format)."""
    import io
    import soundfile as sf
    y, file_sr = sf.read(io.BytesIO(audio_bytes))
    if len(y.shape) > 1:
        y = y.mean(axis=1) 
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    y = y.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    try:
        mfcc = extract_mfcc_from_bytes(audio_bytes)
        tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        return jsonify({
            "prediction": CLASSES[probs.argmax()],
            "confidence": float(probs.max()) * 100,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
