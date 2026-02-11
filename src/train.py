from dataset import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
X,y =load_data #this will load (samples, time, mfcc)
X=X[..., np.newaxis] # this is to add one more dimension to X as CNN accepts 4D inputs only so this will add 1 at end
#First Split is for train + temp
X_train, X_temp, y_train, y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
#Second split is for Validation + Test
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
model=build_model()
model.fit(
    X_train,y_train,validation_data=(X_val,y_val),
    epochs=40, batch_size=32
)
test_loss,test_acc=model.evaluate(X_test,y_test)
print("Test acuracy:",test_acc)