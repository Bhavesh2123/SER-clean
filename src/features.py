import librosa
import numpy as np
from config import SAMPLE_RATE, N_MFCC, MAX_LEN

def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    if mfcc.shape[0] < MAX_LEN:
        pad_widht =MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc,((0, pad_widht),(0,0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    return mfcc