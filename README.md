# CNNを用いた文書類似度
python3系

## 本ファイル
文書データやコーパス入りのファイルは大きすぎてgitにあげられなかったので、dropboxにあげました。ので、実行する場合は[こちら](https://www.dropbox.com/s/a4o9evm1f3yanis/NN_graph.zip?dl=0)からダウンロードしてください。
gitにあげた分は、実装したコードのみです。



## 実行
まずディレクトリ配下に移動して、```pip install -r requirements.txt```あるいは```pip3 install -r requirements.txt```を実行してライブラリをインストールしてください。<br>

次にinit.pyを実行してください。word2vec学習とデータのパースを行います。

そのあとCNN_STS.pyを実行すれば、CNNで学習が行われます。
