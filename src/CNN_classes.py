#-*- coding:utf-8 -*-
# デバッグ用のクラス分類問題にCNN
# こちら実行する前に、news20_corpus.pyを実行しておいてください。
# temp/20news/配下にファイルがあれば大丈夫です。
import tensorflow as tf
from word2vecManager import word2vecManager

import config
import pickle

import glob
import numpy as np
np.set_printoptions(threshold=np.inf)

# 構文解析
import re
import random
from scipy.stats import pearsonr
import pickle

# 定数
# 単語ベクトルのサイズ
EMBEDDING_SIZE = 200
# １フィルター種類に対するフィルターの個数
FILTER_NUM = 128
# 1文書に含まれる単語数（全文書合わせてある）
DOCUMENT_LENGTH = 1000

if __name__ == "__main__":
  methodName = "classes"

  output = open('log/sigmoid-'+methodName+'.txt', mode='w')

  pearsonr_value = 0

  category_list = [
          "alt.atheism",
          "comp.graphics",
          "comp.os.ms-windows.misc",
          "comp.sys.ibm.pc.hardware",
          "comp.sys.mac.hardware",
          "comp.windows.x",
          "misc.forsale",
          "rec.autos",
          "rec.motorcycles",
          "rec.sport.baseball",
          "rec.sport.hockey",
          "sci.crypt",
          "sci.electronics",
          "sci.med",
          "sci.space",
          "soc.religion.christian",
          "talk.politics.guns",
          "talk.politics.mideast",
          "talk.politics.misc",
          "talk.religion.misc"
        ]

  #分類対象のクラスインデクス
  entry_list = [0,3,9,13,17]
  entry_list_category = [category_list[i] for i in entry_list]
  print(len(entry_list),"クラス分類")

  #データを取得。
  #7:3ぐらいでinput,targetに分ける
  #flatに連結
  category_data_input  = []
  category_data_target = []
  for i in entry_list:
    print(category_list[i],"読み込み")
    with open('temp/20news/'+category_list[i], mode='rb') as f:
      docs = pickle.load(f)
      separate_index = int(len(docs)*0.7)
      category_data_input.extend(docs[:separate_index])
      category_data_target.extend(docs[separate_index+1:])


  with tf.Graph().as_default():

    # 変数
    with tf.name_scope('input'): 
      # インプット変数（各文書が　単語数 x 単語ベクトル　のマトリクス）
      x = tf.placeholder(tf.float32, [None, DOCUMENT_LENGTH, EMBEDDING_SIZE], name="x")
      # アウトプット変数5クラス
      y_ = tf.placeholder(tf.float32, [None, len(entry_list)], name="y_")
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
    
    with tf.name_scope('output'):
      # アウトプット層
      class_num = len(entry_list)
      W_o = tf.Variable(tf.truncated_normal([filter_num_total, class_num], stddev=0.1), name="W")
      b_o = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")


      linear = tf.matmul(h_drop, W_o) + b_o
      scores = linear

      #予測は5クラス
      predictions = tf.argmax(scores, 1, name="predictions")

    # 損失関数
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(scores, y_)
      loss = tf.reduce_mean(losses)

      # 正答率
      correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    with tf.name_scope('optimize'):
      # Adamオプティマイザーによるパラメータの最適化
      global_step = tf.Variable(0, name="global_step", trainable=False)
      train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)


    

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # Creates a SummaryWriter
      # Merges all summaries collected in the default graph
      subdir = "sigmoid-"+methodName
      # summaryの設定
      tf.summary.scalar('cross_entropy', loss)
      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter("log/sigmoid-classes/" + subdir, sess.graph)

      # ミニバッチ学習
      for i in range(101):

          # ミニバッチ（100件ランダムで取得）
          # training_xyには、modelsで定義した各文書行列及び正解ラベル（カテゴリ）が入っている
          #{  
          # "tag":"タグ名",
          # "no":tag_number,
          # "doc":"文書",
          # "origin_doc":{"word","word",..},
          # "word2vec_list":{"word2vec","word2vec",...}
          #}
          samples = random.sample(category_data_input, 100)
          batch_xs = np.array([sample["word2vec_list"] for sample in samples])
          
          batch_ys = []
          for sample in samples:
            index = entry_list_category.index(sample["tag"])
            feature = [0]*class_num
            feature[index] = 1
            batch_ys.append(feature)
          batch_ys = np.array(batch_ys)
          #print(batch_ys)

          batch_ys = batch_ys.reshape(len(batch_ys),class_num)#reshapeしろ問題
          # 確率的勾配降下法を使い最適なパラメータを求める
          # dropout_keep_probは0.5を指定
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 0.5})

          if i % 10 == 0:
              # 100件毎に正答率を表示
              a = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 1.0})
              print("TRAINING(%d): %.0f%%" % (i, (a * 100.0)))

              summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 1.0})
              summary_writer.add_summary(summary_str, i)

      # 精度確認
      print("最終精度確認")
      samples = category_data_target          
      batch_xs = np.array([sample["word2vec_list"] for sample in samples])    
      batch_ys = []
      for sample in samples:
        index = entry_list_category.index(sample["tag"])
        feature = [0]*class_num
        feature[index] = 1
        batch_ys.append(feature)
      batch_ys = np.array(batch_ys)
      batch_ys = batch_ys.reshape(len(batch_ys),class_num)#reshapeしろ問題
      a = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 1.0})
      print("TEST DATA ACCURACY: %.0f%%" % (a * 100.0))








