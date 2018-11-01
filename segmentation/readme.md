﻿# 画像分類

## segmentation.py: Fully convolutional network (ここではU-net)を用いて2D画像セグメンテーション

実行準備
segmentation.pyと同じ場所に以下のディレクトリを作成してください．
*train_image
*train_label
*test_image
*test_label
*result

実行方法
```bash
python segmentation.py
```

### コード説明：データの準備
トレーニング用原画像及びラベル画像，テスト用原画像及びラベル画像を各フォルダから読み込みます．原画像とラベル画像の画素値は0から1の実数となるよう正規化します．読み込み結果確認のため，トレーニング用原画像及びラベル画像を数枚表示します．

### コード説明：ネットワークの定義と学習
3つの関数でネットワークを定義しています．
*network_unet：U-net [1]
*network_unet_simple1：U-netの深さを減少させたもの
*network_unet_simple2：U-netの深さをさらに減少させたもの
U-netはネットワークが複雑でCPU実行に時間を要するため，今回は簡易化したU-netであるnetwork_unet_simple2を使用します．ネットワークを変更する場合は
model = network_unet_simple2()
を変更してください．
compileでトレーニングの設定を行い，fitでトレーニングを行います．トレーニング中はトレーニング用原画像とラベル画像を用いてネットワークの学習を行います．さらにバリデーション用データとしてテスト用原画像とラベル画像を使用し，トレーニングデータセットと異なるデータセットに対するネットワークのセグメンテーション性能を確認します．

### コード説明：学習結果の確認
ネットワーク定義及び学習後のネットワークのパラメータをそれぞれファイルに保存します．ネットワーク定義ファイルは人が可読であり，テキストエディタで編集可能です．
トレーニング中の各epochにおける，トレーニング及びバリデーションデータに対する精度と損失関数の値をグラフで表示します．

### コード説明：推定の実行
テスト用原画像を学習済みネットワークに与え，推定ラベル画像を作成します．結果の確認のため，テスト用原画像と推定ラベル画像を数枚表示します．推定ラベル画像をresultディレクトリに保存します．ファイル名はテスト用正解ラベル画像と同じものとしています．

[1] Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015, 9351, pp.234-241, 2015.