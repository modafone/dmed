# -*- coding: utf-8 -*-
"""
JSCAS 2018 handson
Prepare data for superresolution. Generate low resolution images.
@author: Masahiro Oda (Nagoya Univ.), Takeshi Hara (Gifu Univ.)

This is a part of the Deep learning sample code collection for MEDical image processing: DMED
DMED is proposed in:
小田昌宏，原武史，森健策，``医用画像処理のための深層学習サンプルコード集DMED,'' 日本コンピュータ外科学会誌 第27回日本コンピュータ外科学会大会特集号（JSCAS2018）, 2018
"""

import os
import cv2
import numpy as np

IMAGE_IN_DIR = ".\\lidc_train\\"
IMAGE_OUT_DIR = ".\\lidc_train_low\\"
IMAGE_SIZE = 256
IMAGE_SCALE = 3.0

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

# 低解像度画像を作成．画像リスト内の画像を1/scale倍サイズに変更した後元サイズに戻して低解像度画像作成
# imagelist: 画像リスト, scale: 画像サイズ変更係数
def generate_lowresolution(imagelist, scale):
    imagelist_out = []

    for i in range(0, len(image)):
        height, width = imagelist[i].shape[:2]
        image_temp = cv2.resize(imagelist[i], (round(height/scale), round(width/scale)), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
        image_temp = cv2.resize(image_temp, (height, width), interpolation = cv2.INTER_CUBIC)  #主に拡大するのでINTER_CUBIC使用
        imagelist_out.append(image_temp)
    imgsdata = np.asarray(imagelist_out, dtype=np.float32)

    return imgsdata

# 指定ディレクトリに画像リストの画像を保存する
# savepath: ディレクトリ文字列, filenamelist: ファイル名リスト, imagelist: 画像リスト
def save_images(savepath, filenamelist, imagelist):
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        testimage = np.delete(testimage, 2, 1)  # グレースケール画像を保存するときだけ使用．チャンネルに相当する3列目削除
        cv2.imwrite(filename, testimage)


# 画像読み込み
image, image_filenames = load_images(IMAGE_IN_DIR, IMAGE_SIZE, 'Gray')

# 画像解像度変更
image = generate_lowresolution(image, IMAGE_SCALE)

# 画像保存
save_images(IMAGE_OUT_DIR, image_filenames, image)
