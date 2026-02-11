from dataset import load_data
from model import build_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
X,y =load_data("data",augment=False) #this will load (samples, time, mfcc)
X=X[..., np.newaxis] # this is to add one more dimension to X as CNN accepts 4D inputs only so this will add 1 at end
#First Split is for train + temp
X_train, X_temp, y_train, y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
#Second split is for Validation + Test
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
X_aug,y_aug=load_data("data",augment=True)
X_aug=X_aug[...,np.newaxis]
X_train=np.concatenate((X_train,X_aug))
y_train=np.concatenate((y_train,y_aug))
model=build_model()
history=model.fit(
    X_train,y_train,validation_data=(X_val,y_val),
    epochs=40, batch_size=32
)
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
test_loss,test_acc=model.evaluate(X_test,y_test)
print("Test acuracy:",test_acc)
y_pred=model.predict(X_test)
y_pred=np.argmax(y_pred,axis=1)
cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion MAtrix")
plt.show()
