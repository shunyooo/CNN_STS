#-*- coding:utf-8 -*-
import tensorflow as tf
from word2vecManager import word2vecManager
from SemEvalManager import SemEvalManager

import config
import pickle

import glob
import numpy as np
np.set_printoptions(threshold=np.inf)

# 構文解析
import re
import random
from scipy.stats import pearsonr

import seaborn as sns
import pandas as pd

# パースを行うか否か
isParse = True

# 1素性に含まれる単語数の基準長（可変）
DOCUMENT_LENGTH = 50

# 単語ベクトルのサイズ（word2vecで設定しているので、固定で。）
EMBEDDING_SIZE = 200

# ミニバッチのサイズ
MINIBATCH_SIZE = 100

# 素性の作り方(paddingの方法)
# padding-cross 文書を交差して構成
# padding-side  上下に寄せたもの
# padding-down  下にpadding系
# から選択してください
METHOD_NAME = "padding-cross"

#世代数
EPOCH = 1000


if __name__ == "__main__":

  methodName = METHOD_NAME

  # ログ出力用
  output = open('log/sigmoid-'+methodName+'.txt', mode='w')

  pearsonr_value = 0

  # semevalデータから素性を作る。paddingなどを行う。
  semEval = SemEvalManager()

  for i in range(6):
    nameList = [
                "deft-forum_450",
                "deft-news_300",
                "headlines_750",
                "images_750",
                "OnWN_750",
                "tweet-news_750"
    ]

    pathList = [
                "../STS_data/parsed/origined/input/STS.input.deft-forum_450.txt",
                "../STS_data/parsed/origined/input/STS.input.deft-news_300.txt",
                "../STS_data/parsed/origined/input/STS.input.headlines_750.txt",
                "../STS_data/parsed/origined/input/STS.input.images_750.txt",
                "../STS_data/parsed/origined/input/STS.input.OnWN_750.txt",
                "../STS_data/parsed/origined/input/STS.input.tweet-news_750.txt"
                ]

    docPathList = [config.inputDirPath[:-1]+"STS.input."+name+".txt" for name in nameList]
    print(docPathList)

    TARGETNUM = i
    targetName = nameList[TARGETNUM]
    targetPath = pathList[TARGETNUM]
    igPathList = [path for path in pathList if path != targetPath]
    print("\ntarget --> ",targetPath)
    print("train  --> ",igPathList)

    #学習用データ、テストデータの保存先。
    outputInputPath = "temp/pair_word2vec_ans_doc_input_"+methodName+'_'+targetName+".list"
    outputTargetPath = "temp/pair_word2vec_ans_doc_target_"+methodName+'_'+targetName+".list"

    if isParse:
      #文書をword2vec群と類似度の組を漬ける
      #学習データに関して
      print("学習データのパース")
      semEval.transe_word2vec_from_parsedoc(outputPath = outputInputPath,
        ignorePathList = targetPath,methodName = methodName,
        EMBEDDING_SIZE = EMBEDDING_SIZE,DOCUMENT_LENGTH = DOCUMENT_LENGTH)
      #評価データに関して
      print("評価データのパース")
      semEval.transe_word2vec_from_parsedoc(outputPath = outputTargetPath,
        ignorePathList = igPathList,methodName = methodName,
        EMBEDDING_SIZE = EMBEDDING_SIZE,DOCUMENT_LENGTH = DOCUMENT_LENGTH)


    # word2vec群と類似度の組の読み込み
    with open(outputInputPath, mode='rb') as f:
      print("学習データの読み込み")
      result = pickle.load(f)
    print(len(result))
    output.write("学習データ（2素性）"+str(len(result))+"\n")

    # 定数
    # １フィルター種類に対するフィルターの個数
    FILTER_NUM = 128


    with tf.Graph().as_default():

      # 変数
      with tf.name_scope('input'):
        # インプット変数（各文書が　単語数 x 単語ベクトル　のマトリクス）
        x = tf.placeholder(tf.float32, [None, DOCUMENT_LENGTH, EMBEDDING_SIZE], name="x")
        # アウトプット変数（文書類似度 0-5）とりあえず 1で
        y_ = tf.placeholder(tf.float32, [None, 1], name="y_")
        # ドロップアウト変数
        # 出力層の計算を行う時に、プーリング層の結果をランダムに間引くためのもの。過学習を防ぐ。
        # 学習時には0.5,評価の際には１で。
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # インプット変数の次元を拡張しておく（channel）
        # この段階で x は[バッチ数、幅、高さ]の3次元のテンソル。
        # conv2d関数は[バッチ数、幅、高さ、チャネル]の4次元のテンソルを必要とするので、拡張が必要。
        # つまりチャネル数の指定が必要であり、これは画像ではRGBに相当。
        # 文書では、word2vec以外に用いるものがあればチャネルを増やすことで複数の表現を巻き取り可能？
        x_expanded = tf.expand_dims(x, -1)


      # 畳み込み層
      with tf.name_scope('convolution'):
        # フィルタサイズ：3単語、4単語、5単語の３種類のフィルタ
        filter_sizes = [3, 4, 5]
        # 各フィルタ処理結果をMax-poolingした値をアペンドしていく
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # フィルタのサイズ（単語数(高さ), エンベディングサイズ(幅)、チャネル数、フィルタ数）
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, FILTER_NUM]
            # フィルタの重み、バイアス
            W_f = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_f = tf.Variable(tf.constant(0.1, shape=[FILTER_NUM]), name="b")
            # Tensorflowの畳み込み処理
            conv = tf.nn.conv2d(
                x_expanded,
                W_f,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"
            )
            # 活性化関数にはReLU関数を利用
            h = tf.nn.relu(tf.nn.bias_add(conv, b_f), name="relu")

            # プーリング層 Max Pooling
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, DOCUMENT_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool"
            )

            # 各単語数フィルタのプーリング結果が格納される。
            pooled_outputs.append(pooled)

        # プーリング層の結果をつなげる
        # 生成されたfilterの総数。
        filter_num_total = FILTER_NUM * len(filter_sizes)
        # 3次元で結合
        h_pool = tf.concat(3, pooled_outputs)
        # 一つにまとめる。
        h_pool_flat = tf.reshape(h_pool, [-1, filter_num_total])


      # ドロップアウト（トレーニング時0.5、テスト時1.0）
      h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

      # デバッグで行った多クラス分類と素性以外で違うのは、
      # 活性化関数の定義    (類似度:sigmoid*5,多クラス分類:恒等関数)
      # 損失関数の定義     (類似度:二乗誤差   ,多クラス分類:クロスエントロピー)
      # です。
      # バグがあるとすれば、このあたりで予想と違う挙動をしている可能性があります。。。
      with tf.name_scope('output'):
        # アウトプット層
        #class_numは出力の次元数。数値予測より、一つの出力。
        class_num = 1
        #重みとバイアス定義
        W_o = tf.Variable(tf.truncated_normal([filter_num_total, class_num], stddev=0.1), name="W")
        b_o = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")

        # 数値の範囲を0-5に絞るべきでは->sigmoid*5で対処。
        linear = tf.matmul(h_drop, W_o) + b_o
        scores = tf.nn.sigmoid(linear,name = "sigmoid")

      # 損失関数
      with tf.name_scope("loss"):
        #l2_lossは誤差二乗和
        loss = tf.nn.l2_loss(scores - y_/5)

      with tf.name_scope('optimize'):
        # Adamオプティマイザーによるパラメータの最適化
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)


      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # tensorboard用
        # Creates a SummaryWriter
        # Merges all summaries collected in the default graph
        subdir = "sigmoid-"+targetName+"-"+methodName
        # summaryの設定
        tf.summary.scalar('l2loss', loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("log/sigmoid-"+methodName+"/" + subdir, sess.graph)

        # ミニバッチ学習
        for i in range(EPOCH+1):

            # ミニバッチ（100件ランダムで取得）
            # training_xyには、modelsで定義した各文書行列及び正解ラベル（カテゴリ）が入っている
            samples = random.sample(result, MINIBATCH_SIZE)
            batch_xs = np.array([[s[0] for s in samples],[s[1] for s in samples]])
            batch_ys = np.array([s[2] for s in samples])
            batch_ys = batch_ys.reshape(len(batch_ys),1)#reshapeしろ問題

            # 確率的勾配降下法を使い最適なパラメータを求める
            # dropout_keep_probは0.5を指定
            sess.run(train_step, feed_dict={x: batch_xs[0] ,y_: batch_ys, dropout_keep_prob: 0.5})
            sess.run(train_step, feed_dict={x: batch_xs[1] ,y_: batch_ys, dropout_keep_prob: 0.5})

            if i % 10 == 0:
                # 100件毎に正答率を表示
                h,s,a,l = sess.run([h_pool_flat,scores,loss,linear], feed_dict={x: batch_xs[0], y_: batch_ys, dropout_keep_prob: 1.0})

                # tensorboard用
                summary_str = sess.run(summary_op, feed_dict={x: batch_xs[0], y_: batch_ys, dropout_keep_prob: 1.0})
                summary_writer.add_summary(summary_str, i)

                #print("TRAINING(",i,"): ",s,batch_ys,a)
                #print("全結合層",h)
                print("TRAINING(",i,"): ",a)
                r, p = pearsonr(s, batch_ys)
                print("相関:",r," 有意確率",p)

        # 精度確認
        print("最終精度確認")
        with open(outputTargetPath, mode='rb') as f:
          print("評価データの読み込み")
          targets = pickle.load(f)
          print(targetPath)
          #print("確認0行目 ",targets[0])
          batch_xs = np.array([s[0] for s in targets])
          batch_ys = np.array([s[2] for s in targets])
          batch_ys = batch_ys.reshape(len(batch_ys),1)
          print(len(batch_xs))

          # 出力比較ログ
          s,a,l = sess.run([scores,loss,linear], feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 1.0})
          with open("log/log_"+methodName+"_"+targetName+"_"+str(EPOCH)+"epoch.txt", mode='w') as logout:
              logout.write("学習データリスト\n")
              igNameList = [name for name in nameList if name != targetName]
              for name in igNameList:
                logout.write(name+"\n")
              logout.write("\nテストデータ\n")
              logout.write(targetName+"\n\n")
              logout.write("世代数:"+str(EPOCH)+"\n")
              logout.write("学習データサイズ:"+str(len(targets))+"\n")
              logout.write("テストデータサイズ:"+str(len(result))+"\n")
              logout.write("ミニバッチサイズ:"+str(MINIBATCH_SIZE)+"\n")
              logout.write("素性の作り:"+methodName+"\n")

              logout.write("----------------------------------------------------\n\n")
              doc_text = open(docPathList[TARGETNUM],mode='r')
              for ans,out,doc in zip(batch_ys,s,doc_text):
                  logout.write(doc+"正解"+str(ans[0])+", 出力"+str(out[0]*5)+"\n\n")

          # 相関を描画
          df = pd.DataFrame({"ans":[x[0] for x in batch_ys],"out":[y[0] for y in s]})
          sns.jointplot(x="out", y="ans", data=df);
          sns.plt.show()

          print("損失関数",a)
          r, p = pearsonr(s, batch_ys)
          print("相関:",r," 有意確率",p)
          pearsonr_value += r
          output.write(targetPath+"\n 損失関数"+str(a)+"\n 相関:"+str(r)+" 有意確率"+str(p)+"\n\n")

  output.write("\n相関和:"+str(pearsonr_value)+"\n相関平均:"+str(pearsonr_value/6))
