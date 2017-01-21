from word2vecManager import word2vecManager

import config
import pickle

import glob
import numpy as np
np.set_printoptions(threshold=np.inf)

# 構文解析
from treetaggerManager import TRTG_NLP
import re
import random
from scipy.stats import pearsonr


# SemEvalのデータから素性に変換するクラス。
# 文書のパースはcorenlpManagerクラスで済ませている前提。
# 一番下のtranse_word2vec_from_parsedocメソッドが重要。
# padding-...メソッドは、素性を作成する際のpaddingの手法によって分けている。
class SemEvalManager():

  def __init__(self):
    #word2vecの読み込み
    self.model = word2vecManager()
    self.model.load()
    print("wor2vec格納単語数",len(self.model.model.vocab.keys()))


  # word2vecのリスト(文書を表す)から、素性を作成するメソッド。
  # これは、素性を交差させて作成。
  def padding_cross(self,word2vecList1,word2vecList2,max_length,EMBEDDING_SIZE):
    #word2vecの調整。長さを揃える。各max_length/2に揃う？
    if len(word2vecList1) > max_length/2:
      word2vecList1 = word2vecList1[:max_length/2]
    elif len(word2vecList1) < max_length/2:
      word2vecList1 = np.r_[word2vecList1, np.array([[0.0]*EMBEDDING_SIZE]*(int(max_length/2)-len(word2vecList1)))]

    if len(word2vecList2) > max_length/2:
      word2vecList2 = word2vecList2[:max_length/2]
    elif len(word2vecList2) < max_length/2:
      word2vecList2 = np.r_[word2vecList2, np.array([[0.0]*EMBEDDING_SIZE]*(int(max_length/2)-len(word2vecList2)))]       

    feature1 = []
    for i in range(max(len(word2vecList1),len(word2vecList2))):
      if i < len(word2vecList1):
        feature1.append(word2vecList1[i])
      if i < len(word2vecList2):
        feature1.append(word2vecList2[i])
    feature1 = np.array(feature1)

    feature2 = []
    for i in range(max(len(word2vecList1),len(word2vecList2))):
      if i < len(word2vecList1):
        feature2.append(word2vecList2[i])
      if i < len(word2vecList2):
        feature2.append(word2vecList1[i])
    feature2 = np.array(feature2)

    '''print("feature1")
    print_feature(feature1)
    print("feature2")
    print_feature(feature2)'''

    return feature1,feature2

  # word2vecのリスト(文書を表す)から、素性を作成するメソッド。
  # これは、素性を上下に寄せて作成。
  # 超過した場合は、それぞれの文書を下から削り揃える。この場合でも最低区切り線は挿れる。
  def padding_side(self,word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE):
    #マージンベクトル。
    doc_length = len(word2vecList1)+len(word2vecList2)
    #print(len(word2vecList1),len(word2vecList2))

    if max_length > doc_length:
      margin = np.array([[0.0]*EMBEDDING_SIZE]*(max_length-doc_length))
      #print("不足")
      #print(len(word2vecList1),len(margin),len(word2vecList2))

    else:
      #各文書切るindexの設定
      line_index = (max_length-margin_size)/2
      word2vecList1 = word2vecList1[:line_index]
      word2vecList2 = word2vecList2[:line_index]
      doc_length = len(word2vecList1)+len(word2vecList2)
      margin = np.array([[0.0]*EMBEDDING_SIZE]*(max_length-doc_length))
      #print("超過")
      #print("line_index",line_index)
      #print(len(word2vecList1),len(margin),len(word2vecList2))

    feature1 = np.r_[word2vecList1,margin,word2vecList2]
    feature2 = np.r_[word2vecList2,margin,word2vecList1]

    '''print("feature1")
    print_feature(feature1)
    print("feature2")
    print_feature(feature2)'''

    return np.array(feature1),np.array(feature2)

  # word2vecのリスト(文書を表す)から、素性を作成するメソッド。
  # これは、素性の区切り線を中央(max_length/2)に揃えるもの。
  # 基準長に合わせる実装はまだしていません...
  def padding_center(self,word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE):
    print("padding_centerの基準長verは未実装。違う方法を選んでください。")
    exit()
    pass

  # word2vecのリスト(文書を表す)から、素性を作成するメソッド。
  # これは、素性をmargin_sizeの大きさの区切り線で区切り、
  # 基準長max_lengthに合わせる一番愚直な方法。
  # 基準長に足りない場合は下をpadding。
  # 基準長を超過した場合、それぞれの文書を下から削り揃える。この場合でも最低区切り線は挿れる。
  def padding_down(self,word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE):

    doc_length = len(word2vecList1)+margin_size+len(word2vecList2)

    #超過した場合、各文書を削る。
    if max_length < doc_length:
      #print("超過")
      #各文書切るindexの設定
      line_index = (max_length-margin_size)/2
      word2vecList1 = word2vecList1[:line_index]
      word2vecList2 = word2vecList2[:line_index]

    #足りない部分を埋める
    doc_length = len(word2vecList1)+margin_size+len(word2vecList2)
    padding = np.array([[0.0]*EMBEDDING_SIZE]*(max_length - doc_length))
    margin = np.array([[0.0]*EMBEDDING_SIZE]*(margin_size))

    if len(padding) > 0:
      feature1 = np.r_[word2vecList1,margin,word2vecList2,padding]
      feature2 = np.r_[word2vecList2,margin,word2vecList1,padding]
    else:
      #print(padding)
      #print(len(word2vecList1),len(margin),len(word2vecList2))
      feature1 = np.r_[word2vecList1,margin,word2vecList2]
      feature2 = np.r_[word2vecList2,margin,word2vecList1]
      
    #print(feature1.shape)
    '''print("feature1")
    print_feature(feature1)
    print("feature2")
    print_feature(feature2)'''

    return np.array(feature1),np.array(feature2)

  #素性確認用。
  def print_feature(self,feature,num = 3):
    print("[")
    for vec in feature:
      print("[",end = "")
      for i in range(num):
        print(vec[i],end = ",")
      print("]")
    print("]")


  ##########################################################################
  #######パース文書、回答類似度を読み込み、文書をword2vec群と類似度の組に。########
  ##########################################################################
  #[
  #[[word2vec,word2vec,...],[word2vec,word2vec,word2vec,...],類似度],
  #[[word2vec,word2vec,...],[word2vec,word2vec,word2vec,...],類似度],
  #[[word2vec,word2vec,...],[word2vec,word2vec,word2vec,...],類似度],
  #]
  # 何回もやらないので、outputPathに保存(pickle)する。
  # margin_size         :区切り線に用いる単語数の幅
  # gsDirPath           :類似度回答データ
  # parsedInputDirPath  :パースされた文書が格納されているディレクトリ
  # outputPath          :pickle保存する先Path
  # ignorePathList      :格納されている文書のなかで、無視するリスト。(学習データとテストデータを分けるため)
  # EMBEDDING_SIZE      :単語のベクトルのサイズ。paddingするときに必要。
  # DOCUMENT_LENGTH     :文書の基準長。足りない場合はpadding,超過した場合は削り、長さを合わせる。
  # methodName          :素性の作り方、paddingの方法を指定する。
  def transe_word2vec_from_parsedoc(self,margin_size = 3,
    gsDirPath = config.gsDirPath,
    parsedInputDirPath = config.parsedInputDirPath,
    outputPath = "temp/pair_word2vec_ans.list",
    ignorePathList = None, 
    EMBEDDING_SIZE = 200,
    DOCUMENT_LENGTH = 30,
    methodName = "padding-cross"):


    #半角英字のみを抽出する正規表現->パースに用いる。
    lower_reg = re.compile(r'^[a-z]+$')

    # word2vecと類似度の組みを格納するもの。これを返却する。
    result = []
    print("ignore->",ignorePathList)
    print("parse start")
    #それぞれのパスに関して学習。
    #一行毎にタブ区切り２文書、正解類似度となっている。
    for (gsPath,inputPath) in zip(glob.glob(gsDirPath),glob.glob(parsedInputDirPath)):
      print(inputPath)

      if inputPath in ignorePathList:
        print('↑ ignore')
        continue

      #入力、回答ファイル毎に評価
      with open(gsPath,'r') as ansFile, open(inputPath,'r') as inputFile:
        #一行毎にword2vec、類似度の組みを作成。
        for ans,pair_doc in zip(ansFile,inputFile):
          pair_doc = pair_doc[:-1].split('\t') 
          #print(pair_doc,ans[:-1])
          if len(pair_doc) == 2:
            #スペース区切り、記号、数字を含むものは除外
            docList1 = [word for word in pair_doc[0].split(' ') if lower_reg.search(word)]
            word2vecList1 = self.model.getWord2VecByList(docList1)
            docList2 = [word for word in pair_doc[1].split(' ') if lower_reg.search(word)]
            word2vecList2 = self.model.getWord2VecByList(docList2)

          #最大長を決定。別々にやった場合、学習データと評価データに差が出るので、統一。
          max_length = DOCUMENT_LENGTH


          # 素性の作成。ここで素性の作り方を工夫。
          if methodName == "padding-cross":
            feature1,feature2 = self.padding_cross(word2vecList1,word2vecList2,max_length,EMBEDDING_SIZE)
          elif methodName == "padding-side":
            feature1,feature2 = self.padding_side(word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE)
          elif methodName == "padding-center":
            feature1,feature2 = self.padding_center(word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE)
          elif methodName == "padding-down":
            feature1,feature2 = self.padding_down(word2vecList1,word2vecList2,max_length,margin_size,EMBEDDING_SIZE)
          else:
            print("methodNameが不正です。")
            exit()

          result.append([feature1,feature2,float(ans),pair_doc])

    print(len(word2vecList1),len(word2vecList2))  
    print(feature1.shape)
    print(feature2.shape)
    print("parse is done.")
    print("max_length = ",max_length)

    #保存。何回もやっては重いので。
    with open(outputPath,'wb') as output:
      pickle.dump(result,output)
      print("pickled result to ",outputPath)

    print()
    #一応返しておく
    return result