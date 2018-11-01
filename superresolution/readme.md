# 超解像

## superresolution.py: 2D画像超解像

実行準備
superresolution.pyと同じ場所に以下のディレクトリを作成してください．
* lidc_train：トレーニング用原画像
* lidc_train_low：トレーニング用低解像度画像
* lidc_test：テスト用原画像
* lidc_test_low：テスト用低解像度画像
* result：推定結果

原画像から画像処理によって低解像度画像を作りだし，超解像の実験を行います．高解像度と低解像度画像のペアが既にある場合は，低解像度画像を作る処理は飛ばしてください．

トレーニング用の原画像を用意し，lidc_trainディレクトリに格納してください．画像はbmp, jpg, pngのどれかの形式としてください．テスト用原画像も同様に用意し，lidc_testディレクトリに格納してください．

トレーニング用の原画像からトレーニング用低解像度画像を作ります．次のように実行してください．
```bash
python superresolution_prepare.py
```

次に，superresolution_prepare.pyの
```bash
 IMAGE_IN_DIR = ".\\lidc_train\\"
 IMAGE_OUT_DIR = ".\\lidc_train_low\\"
```
の部分を
```bash
 IMAGE_IN_DIR = ".\\lidc_test\\"
 IMAGE_OUT_DIR = ".\\lidc_test_low\\"
```
と書き換えてから再度superresolution_prepare.pyを実行してください．

ソースコード中のパラメータは次の意味です．
* IMAGESIZE = 256：画像を読み込んでIMAGESIZE×IMAGESIZE 画素のサイズにリサイズしてから処理を行う．
* EPOCHS = 3：学習の反復回数．

画像が用意できたら超解像処理を実行します．
```bash
python superresolution.py
```

終了後に，resultディレクトリにlidc_test画像からの超解像結果が保存されています．

### コード説明：概要
粗い画像から原画像を復元します．原画像だけが提供されるので，まず実験のために粗い画像を作成します（superresolution_prepare.py使用）．その後超解像ネットワークの学習及び推定を行います（superresolution.py使用）．

### superresolution_prepare.pyの解説
IMAGE_IN_DIRで指定されたディレクトリ内の全ての画像を読み込み，画像をIMAGE_SCALE分の1のサイズに縮小（平均値補間使用）します．そして元サイズに拡大します（バイキュービック補間使用）．画像をIMAGE_OUT_DIRで指定されたディレクトリに保存します．

### superresolution.pyの解説
#### コード説明：データの準備
トレーニング用原画像及び低解像度画像，テスト用原画像及び低解像度画像を各フォルダから読み込みます．画像の画素値は0から1の実数となるよう正規化します．

#### コード説明：ネットワークの定義と学習
2つの関数でネットワークを定義しています．
* network_srcnn：Super Resolution CNN (SRCNN) [1]
* network_ddsrcnn：Deep Denoising Super Resolution CNN (DDSRCNN) [2]

使用するネットワークを変更する場合は
model = network_srcnn()
を変更してください．

学習中の精度評価には，超解像では画質再現性の尺度となるPeak Signal-to-Noise Ratio（PSNR）が使われます．PSNRを計算する評価関数はKerasに用意されていないので関数psnrとして定義しています．

compileで評価関数，損失関数を含むトレーニングの設定を行い，fitでトレーニングを行います．トレーニング中はトレーニング用低解像度画像と原画像を用いてネットワークの学習を行います．さらにバリデーション用データとしてテスト用低解像度画像と原画像を使用し，トレーニングデータセットと異なるデータセットに対する推定結果の良し悪しを確認します．

#### コード説明：学習結果の確認
ネットワーク定義及び学習後のネットワークのパラメータをそれぞれファイルに保存します．ネットワーク定義ファイルは人が可読であり，テキストエディタで編集可能です．
トレーニング中の各epochにおける，トレーニング及びバリデーションデータに対する評価関数の値と損失関数の値をグラフで表示します．

#### コード説明：推定の実行
テスト用低解像度画像を学習済みネットワークに与え，推定画像を作成します．結果の確認のため，テスト用低解像度画像と推定画像を数枚表示します．推定画像をresultディレクトリに保存します．ファイル名はテスト用低解像度画像と同じものとしています．

[1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014

[2] Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang. Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections, arXiv:1606.08921, 2016
