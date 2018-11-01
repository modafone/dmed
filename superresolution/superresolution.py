# -*- coding: utf-8 -*-
"""
JSCAS 2018 handson
Image superresolution using SRCNN and DDSRCNN
@author: Masahiro Oda (Nagoya Univ.), Takeshi Hara (Gifu Univ.)

This is a part of the Deep learning sample code collection for MEDical image processing: DMED
DMED is proposed in:
小田昌宏，原武史，森健策，``医用画像処理のための深層学習サンプルコード集DMED,'' 日本コンピュータ外科学会誌 第27回日本コンピュータ外科学会大会特集号（JSCAS2018）, 2018

Super Resolution CNN is proposed in:
Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014

Deep Denoising Super Resolution CNN is proposed in:
Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang. Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections, arXiv:1606.08921, 2016
"""

import os
import cv2
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from keras.models import Model
import keras.optimizers as optimizers
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
EPOCHS = 3


# ディレクトリ内の画像を読み込む
# inputpath: ディレクトリ文字列，imagesize: 画像サイズ, type_color: ColorかGray
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
                testimage = np.asarray(testimage, dtype=np.float64)
                # チャンネル，高さ，幅に入れ替え．data_format="channels_first"を使うとき必要
                #testimage = testimage.transpose(2, 0, 1)
                # チャンネルをbgrの順からrgbの順に変更
                testimage = testimage[::-1]
            
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
        testimage = np.delete(testimage, 2, 1)  # グレースケール画像を保存するときだけ使用．チャンネルに相当する3列目削除
        cv2.imwrite(filename, testimage)


#%%
# データ準備
# 画像読み込み
# training用の低解像度画像と高解像度画像読み込み
imagelow_train, imagelow_train_filenames = load_images(".\\lidc_train_low\\", IMAGE_SIZE, 'Gray')
imagehigh_train, imagehigh_train_filenames = load_images(".\\lidc_train\\", IMAGE_SIZE, 'Gray')
# test用の低解像度画像と高解像度画像読み込み
imagelow_test, imagelow_test_filenames = load_images(".\\lidc_test_low\\", IMAGE_SIZE, 'Gray')
imagehigh_test, imagehigh_test_filenames = load_images(".\\lidc_test\\", IMAGE_SIZE, 'Gray')

# 画素値0-1正規化
imagelow_train /= 255.0
imagehigh_train /= 255.0
imagelow_test /= 255.0
imagehigh_test /= 255.0


#%%
# ネットワークの定義
# Super Resolution CNN (SRCNN)
def network_srcnn():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    
    x = Conv2D(64, (9, 9), activation="relu", padding='same')(input_img)
    x = Conv2D(32, (1, 1), activation="relu", padding='same')(x)
    x = Conv2D(1, (5, 5), padding='same')(x)

    model = Model(input_img, x)
    
    return model

# Deep Denoising Super Resolution CNN (DDSRCNN)
def network_ddsrcnn():
    input_img = Input((IMAGE_SIZE, IMAGE_SIZE, 1), dtype='float')
    
    enc1 = Conv2D(64, kernel_size=3, activation="relu", padding='same')(input_img)
    enc1 = Conv2D(64, kernel_size=3, activation="relu", padding='same')(enc1)
    down1 = MaxPooling2D(pool_size=2)(enc1)
    
    enc2 = Conv2D(128, kernel_size=3, activation="relu", padding='same')(down1)
    enc2 = Conv2D(128, kernel_size=3, activation="relu", padding='same')(enc2)
    down2 = MaxPooling2D(pool_size=2)(enc2)
    
    enc3 = Conv2D(256, kernel_size=3, activation="relu", padding='same')(down2)

    up3 = UpSampling2D(size=2)(enc3)
    dec3 = Conv2D(128, kernel_size=3, activation="relu", padding='same')(up3)
    dec3 = Conv2D(128, kernel_size=3, activation="relu", padding='same')(dec3)
    
    add2 = Add()([dec3, enc2])
    up2 = UpSampling2D(size=2)(add2)
    dec2 = Conv2D(64, kernel_size=3, activation="relu", padding='same')(up2)
    dec2 = Conv2D(64, kernel_size=3, activation="relu", padding='same')(dec2)
    
    add1 = Add()([dec2, enc1])
    dec1 = Conv2D(1, kernel_size=5, activation="linear", padding='same')(add1)
    
    model = Model(input=input_img, output=dec1)
    
    return model

model = network_srcnn()

# ネットワークを表示
print(model.summary())


#%%
def psnr(y_true, y_pred):
    return -10*K.log(K.mean(K.flatten((y_true - y_pred))**2))/np.log(10)
    
# trainingの設定
adam = optimizers.Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[psnr])

# training実行
# training用低解像度画像と高解像度画像でtrainingする．validation用にtest用の低解像度画像と高解像度画像を使用する．
training = model.fit(imagelow_train, imagehigh_train,
                     epochs=EPOCHS, batch_size=10, shuffle=True, validation_data=(imagelow_test, imagehigh_test), verbose=1)


#%%
# training結果をファイルに保存
# ネットワーク定義保存
json_string = model.to_json()
open("keras_srcnn_model_json", "w").write(json_string)
# 重み
model.save_weights('keras_srcnn_weight_hdf5')

# 学習履歴グラフ表示
def plot_history(history):
    plt.plot(history.history['psnr'])
    plt.plot(history.history['val_psnr'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['psnr', 'val_psnr'], loc='lower right')
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
# test用低解像度画像を用いた推定の実行
results = model.predict(imagelow_test, verbose=1)

# 推定結果の画像表示
n = 5
plt.figure(figsize=(25, 10))
for i in range(n):
   # 低解像度画像
   ax = plt.subplot(2, n, i+1)
   plt.imshow(imagelow_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   
   # 推定結果画像
   ax = plt.subplot(2, n, i+1+n)
   plt.imshow(results[i].reshape(IMAGE_SIZE, IMAGE_SIZE))
   plt.gray()
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
plt.show()

# 推定結果の画像をファイル保存
results *= 255.0    # 0-1の範囲の値が出力されるので見やすいように255倍する
save_images(".\\result\\", imagelow_test_filenames, results)
