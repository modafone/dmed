# -*- coding: utf-8 -*-
"""
JSCAS 2018 handson
Image segmentation using U-net
@author: Masahiro Oda (Nagoya Univ.), Takeshi Hara (Gifu Univ.)

This is a part of the Deep learning sample code collection for MEDical image processing: DMED
DMED is proposed in:
小田昌宏，原武史，森健策，``医用画像処理のための深層学習サンプルコード集DMED,'' 日本コンピュータ外科学会誌 第27回日本コンピュータ外科学会大会特集号（JSCAS2018）, 2018

U-net is proposed in:
Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015, 9351, pp.234-241, 2015.
"""

import os
import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
import matplotlib.pyplot as plt

IMAGE_TRAIN_DIR = ".\\train_image\\"
LABEL_TRAIN_DIR = ".\\train_label\\"
IMAGE_TEST_DIR = ".\\test_image\\"
LABEL_TEST_DIR = ".\\test_label\\"
IMAGESIZE = 64
EPOCHS = 15


# ディレクトリ内の画像を読み込む
# inputpath: ディレクトリ文字列, imagesize: 画像サイズ, type_color: ColorかGray
def load_images(inputpath, imagesize, type_color):
    imglist = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                # カラー画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # 色チャンネルをbgrの順からrgbの順に変更
                testimage = cv2.cvtColor(testimage, cv2.COLOR_BGR2RGB)
                testimage = np.asarray(testimage, dtype=np.float64)
                # 色チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
                #testimage = testimage.transpose(2, 0, 1)
            
            elif type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((1, imagesize, imagesize))
                # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
                testimage = testimage.transpose(1, 2, 0)

            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, sorted(files)  # 画像リストとファイル名のリストを返す

# 指定ディレクトリに画像リストの画像を保存する
# savepath: ディレクトリ文字列, filenamelist: ファイル名リスト, imagelist: 画像リスト
def save_images(savepath, filenamelist, imagelist):
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        testimage = np.delete(testimage, 2, 1) # グレースケール画像を保存するときだけ使用．チャンネルに相当する3列目削除
        cv2.imwrite(filename, testimage)


#%%
# データ準備
# 画像読み込み
# training用の原画像とラベル画像読み込み
image_train, image_train_filenames = load_images(IMAGE_TRAIN_DIR, IMAGESIZE, 'Gray')
label_train, label_train_filenames = load_images(LABEL_TRAIN_DIR, IMAGESIZE, 'Gray')
# test用の原画像とラベル画像読み込み
image_test, image_test_filenames = load_images(IMAGE_TEST_DIR, IMAGESIZE, 'Gray')
label_test, label_test_filenames = load_images(LABEL_TEST_DIR, IMAGESIZE, 'Gray')

# 画素値0-1正規化
image_train /= np.max(image_train)
label_train /= np.max(label_train)
image_test /= np.max(image_test)
label_test /= np.max(label_test)


# training用の原画像とラベル画像表示
n = 20
plt.figure(figsize=(40, 4))
for i in range(n):
   # 原画像
   ax = plt.subplot(2, n, i+1)
   plt.imshow(image_train[i].reshape(IMAGESIZE, IMAGESIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   
   # ラベル画像
   ax = plt.subplot(2, n, i+1+n)
   plt.imshow(label_train[i].reshape(IMAGESIZE, IMAGESIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
plt.show()


#%%
# ネットワークの定義
# U-net
def network_unet():
    input_img = Input(shape=(IMAGESIZE, IMAGESIZE, 1))

    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(input_img)
    enc1 = BatchNormalization()(enc1)
    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1)
    
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(down1)
    enc2 = BatchNormalization()(enc2)
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    down2 = MaxPooling2D(pool_size=2, strides=2)(enc2)

    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(down2)
    enc3 = BatchNormalization()(enc3)
    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    down3 = MaxPooling2D(pool_size=2, strides=2)(enc3)
    
    enc4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(down3)
    enc4 = BatchNormalization()(enc4)
    enc4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(enc4)
    enc4 = BatchNormalization()(enc4)
    down4 = MaxPooling2D(pool_size=2, strides=2)(enc4)
    
    enc5 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(down4)
    enc5 = BatchNormalization()(enc5)
    enc5 = Conv2D(1024, kernel_size=3, strides=1, activation="relu", padding="same")(enc5)
    enc5 = BatchNormalization()(enc5)

    up4 = UpSampling2D(size=2)(enc5)
    dec4 = concatenate([up4, enc4], axis=-1)
    dec4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    dec4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(dec4)
    dec4 = BatchNormalization()(dec4)
    
    up3 = UpSampling2D(size=2)(dec4)
    dec3 = concatenate([up3, enc3], axis=-1)
    dec3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)
    dec3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(dec3)
    dec3 = BatchNormalization()(dec3)

    up2 = UpSampling2D(size=2)(dec3)
    dec2 = concatenate([up2, enc2], axis=-1)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    
    up1 = UpSampling2D(size=2)(dec2)
    dec1 = concatenate([up1, enc1], axis=-1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    
    dec1 = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid", padding="same")(dec1)
    
    model = Model(input=input_img, output=dec1)
    
    return model

# U-net簡易版1
def network_unet_simple1():
    input_img = Input(shape=(IMAGESIZE, IMAGESIZE, 1))

    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(input_img)
    enc1 = BatchNormalization()(enc1)
    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1)
    
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(down1)
    enc2 = BatchNormalization()(enc2)
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)
    down2 = MaxPooling2D(pool_size=2, strides=2)(enc2)

    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(down2)
    enc3 = BatchNormalization()(enc3)
    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same")(enc3)
    enc3 = BatchNormalization()(enc3)
    
    up2 = UpSampling2D(size=2)(enc3)
    dec2 = concatenate([up2, enc2], axis=-1)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(dec2)
    dec2 = BatchNormalization()(dec2)
    
    up1 = UpSampling2D(size=2)(dec2)
    dec1 = concatenate([up1, enc1], axis=-1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    
    dec1 = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid", padding="same")(dec1)
    
    model = Model(input=input_img, output=dec1)
    
    return model

# U-net簡易版2
def network_unet_simple2():
    input_img = Input(shape=(IMAGESIZE, IMAGESIZE, 1))

    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(input_img)
    enc1 = BatchNormalization()(enc1)
    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(enc1)
    enc1 = BatchNormalization()(enc1)
    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1)
    
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(down1)
    enc2 = BatchNormalization()(enc2)
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same")(enc2)
    enc2 = BatchNormalization()(enc2)

    up1 = UpSampling2D(size=2)(enc2)
    dec1 = concatenate([up1, enc1], axis=-1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = BatchNormalization()(dec1)
    
    dec1 = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid", padding="same")(dec1)
    
    model = Model(input=input_img, output=dec1)
    
    return model


model = network_unet_simple2()

# ネットワークを表示
print(model.summary())


#%%
# trainingの設定
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# training実行
# training用原画像とラベル画像でtrainingする．validation用にtest用の原画像とラベル画像を使用する．
training = model.fit(image_train, label_train,
                     epochs=EPOCHS, batch_size=5, shuffle=True, validation_data=(image_test, label_test), verbose=1)


#%%
# training結果をファイルに保存
# ネットワーク定義保存
json_string = model.to_json()
open("keras_unet_model_json", "w").write(json_string)
# 重み
model.save_weights('keras_unet_weight_hdf5')

# 学習履歴グラフ表示
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()
    
plot_history(training)


#%%
# test画像を用いた推定の実行
results = model.predict(image_test, verbose=1)

# 推定結果の画像表示
n = 20
plt.figure(figsize=(40, 4))
for i in range(n):
   # 原画像
   ax = plt.subplot(2, n, i+1)
   plt.imshow(image_test[i].reshape(IMAGESIZE, IMAGESIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   
   # 推定ラベル画像
   ax = plt.subplot(2, n, i+1+n)
   plt.imshow(results[i].reshape(IMAGESIZE, IMAGESIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
plt.show()

# 推定結果をファイル保存
results *= 255.0    # 0-1の範囲の値が出力されるので見やすいように255倍する
save_images(".\\result\\", label_test_filenames, results)
