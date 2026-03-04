"""
Pre-compute MFCCs for SpeechCommands dataset and save as memory-mapped files.
Run this script ONCE in the terminal before using the notebook.
"""
import os
import gc
import numpy as np
import librosa
from tqdm import tqdm

ROOT = "./data/SpeechCommands/speech_commands_v0.02"
CACHE_DIR = os.path.join(ROOT, "mfcc_cache")
MAX_LEN = 100
N_MFCC = 40

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc.T  # (100, 40)

def main():
    classes = sorted([
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_")
    ])

    files = []
    labels = []
    for i, c in enumerate(classes):
        folder = os.path.join(ROOT, c)
        for f in os.listdir(folder):
            if f.endswith(".wav"):
                files.append(os.path.join(folder, f))
                labels.append(i)

    n = len(files)
    print(f"Found {n} files across {len(classes)} classes")

    os.makedirs(CACHE_DIR, exist_ok=True)
    mfcc_path = os.path.join(CACHE_DIR, "mfccs.dat")
    label_path = os.path.join(CACHE_DIR, "labels.npy")
    meta_path = os.path.join(CACHE_DIR, "meta.npy")

    # Write MFCCs directly to disk via memmap
    shape = (n, MAX_LEN, N_MFCC)
    fp = np.memmap(mfcc_path, dtype='float32', mode='w+', shape=shape)

    for i in tqdm(range(n), desc="Extracting MFCCs"):
        fp[i] = extract_mfcc(files[i])
        # Flush and GC periodically to keep memory low
        if i % 2000 == 0:
            fp.flush()
            gc.collect()

    fp.flush()
    del fp
    gc.collect()

    np.save(label_path, np.array(labels, dtype=np.int64))
    np.save(meta_path, np.array(shape, dtype=np.int64))  # save shape for loading

    # Also save file list and classes for reference
    np.save(os.path.join(CACHE_DIR, "files.npy"), np.array(files))
    np.save(os.path.join(CACHE_DIR, "classes.npy"), np.array(classes))

    print(f"Done! Saved to {CACHE_DIR}")
    print(f"  mfccs.dat: {os.path.getsize(mfcc_path) / 1e9:.2f} GB")
    print(f"  Shape: {shape}")

if __name__ == "__main__":
    main()
