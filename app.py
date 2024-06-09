from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.utils import to_categorical
import tensorflow
import keras
import numpy as np

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["dog", "cat"]
num_classes = len(classes)
image_size = 64
num_testdata = 25

X_train = []
X_test = []
y_train = []
y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index)

                img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trains)
                X_train.append(data)
                y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")

# /* 学習データ保存 */
np.savez("./dog_cat.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

 # /* 学習データを読み込む関数 */
def load_data():
    loaded_arrays = np.load("./dog_cat.npz", allow_pickle = True)
    X_train = loaded_arrays["X_train"]
    X_test = loaded_arrays["X_test"]
    y_train = loaded_arrays["y_train"]
    y_test = loaded_arrays["y_test"]

    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return X_train, y_train, X_test, y_test

# /* モデルを学習する関数 */
def train(X, y, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.45))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # 修正: lr を learning_rate に変更し、 decay を削除
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(X, y, batch_size=28, epochs=40)

    return model

# /* メイン関数、データの読み込みとモデルの学習を行います。 */
def main():
    X_train, y_train, X_test, y_test = load_data()
    model = train(X_train, y_train, X_test, y_test)
    model.save("cnn.h5")

main()