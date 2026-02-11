from keras.models import Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization)
from keras.optimizers import Adam
from config import MAX_LEN, N_MFCC
def build_model(num_classes=8):
    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),
                     activation="relu",
                     input_shape=(MAX_LEN, N_MFCC, 1)))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3,3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes,activation="softmax"))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],)
    return model

if __name__=="__main__":
    model= build_model()
    model.summary()
