from dataset import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
X,y =load_data
X=X[..., np.newaxis]
X_train, X_temp, y_train, y_temp=train_test_split()