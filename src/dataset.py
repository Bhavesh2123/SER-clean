import os
import numpy as np
from features import extract_mfcc
emotion_map={"01": 0, "02": 1, "03": 2, "04": 3,
    "05": 4, "06": 5, "07": 6, "08": 7}
def load_data(data_dir):
    X, y=[],[]
    for root, _, files in os.wall(data_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion= file.split("-")[2]
                label = emotion_map[emotion]
                path = os.path.join(root, file)
                mfcc = extract_mfcc(path)
                X.append(mfcc)
                y.append(label)
    return np.array(X), np.array(y)