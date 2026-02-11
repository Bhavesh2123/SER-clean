import librosa
import numpy as np
from config import SAMPLE_RATE, N_MFCC, MAX_LEN
import random

def add_noise(signal, noise_factor=0.005):
    noise=np.random.randn(len(signal))
    augmented=signal+noise_factor*noise
    return augmented
def time_stretch(signal, rate=1.0):
    return librosa.effects.time_stretch(signal,rate=rate)
def pitch_shift(signal, sr, n_steps=0):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

def extract_mfcc(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if augment:
        choice=random.choice(["noise","stretch","pitch",None])
        if choice=="noise":
            signal=add_noise(signal)
        elif choice=="stretch":
            rate=random.uniform(0.8,1.2)
            signal=time_stretch(signal,rate)
        elif choice=="pitch":
            steps=random.randit(-2,2)
            signal=pitch_shift(signal,sr,steps)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    if mfcc.shape[0] < MAX_LEN:
        pad_widht =MAX_LEN - mfcc.shape[0]
        mfcc = np.pad(mfcc,((0, pad_widht),(0,0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]
    return mfcc