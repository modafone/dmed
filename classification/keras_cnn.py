# -*- coding: utf-8 -*-
"""
JSCAS 2018 handson
MNIST data classification using convolutional neural network
@author: Masahiro Oda (Nagoya Univ.)

This is a part of the Deep learning sample code collection for MEDical image processing: DMED
DMED is proposed in:
小田昌宏，原武史，森健策，``医用画像処理のための深層学習サンプルコード集DMED,'' 日本コンピュータ外科学会誌 第27回日本コンピュータ外科学会大会特集号（JSCAS2018）, 2018
"""

from keras.datasets import mnist
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten


# mnistデータのロード（インターネットから）
(data_train, label_train), (data_test, label_test) = mnist.load_data()

# 画像配列の形式を変換．色チャンネル数に対応した1次元を追加．(画像数,28,28) -> (画像数,28,28,1)
data_train = np.asarray(data_train, dtype=np.float32).reshape((len(data_train), 28, 28, 1))
data_test = np.asarray(data_test, dtype=np.float32).reshape((len(data_test), 28, 28, 1))

label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)

# 画像の画素値0-1に正規化
data_train /= np.max(data_train)
data_test /= np.max(data_test)

# 正解ラベルをone hot vector形式に変換
label_test_binary = to_categorical(label_test)
label_train_binary = to_categorical(label_train)


#%%
# ネットワーク構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='valid', data_format="channels_last", input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 学習
training = model.fit(data_train, label_train_binary,
                     epochs=10, batch_size=100, verbose=1)


# 評価
results = list(model.predict_classes(data_test, verbose=1))


# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print()
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)
