# -*- coding: utf-8 -*-
"""
JSCAS 2018 handson
MNIST data classification using neural network
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
from keras.layers import Dense, Activation, Dropout

# 2次元データの集合を1次元データの集合に変換
# (data number, width, height) が与えられると (data number, width*height) に変換する
def to_1darray(data):
    datanum, datax, datay = data.shape[:3]
    data = np.reshape(data, (datanum, datax * datay))
    return data


# mnistデータのロード（インターネットから）
(data_train, label_train), (data_test, label_test) = mnist.load_data()

data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)

# 画像を1次元化
data_train = to_1darray(data_train)
data_test = to_1darray(data_test)

# 正解ラベルをone hot vector形式に変換
label_test_binary = to_categorical(label_test)
label_train_binary = to_categorical(label_train)


#%%
# ネットワーク構築
model = Sequential()

model.add(Dense(200, input_dim=784))
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
                     epochs=50, batch_size=100, verbose=1)


# 評価
results = list(model.predict_classes(data_test, verbose=1))


# %%
# 認識率を計算
score = accuracy_score(label_test, results)
print()
print(score)
cmatrix = confusion_matrix(label_test, results)
print(cmatrix)

