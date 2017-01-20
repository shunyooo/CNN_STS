# coding: utf-8
## 	@package word2vec管理用パッケージ。
#	保存、ロードや学習、素性化など。
from gensim import models
import logging
import pickle
import glob
import config

# 構文解析
from treetaggerManager import TRTG_NLP

import numpy as np

## word2vec管理クラス。
class word2vecManager():
	# path      : 対象文書のpath
	# size 		: 出力するベクトル次元数
	# min_count : この数値より低い出現回数の単語は無視
	# window    : 一つの単語に関してこの数値文だけ前後をチェック
	def __init__(self,path = '../STS_data/empty.txt',size=200,min_count=0,window=15,sentences=[]):
		if len(sentences)>0:
			sentences = sentences
		else:
			sentences = models.word2vec.Text8Corpus(path)

		self.tagger = TRTG_NLP()
		self.model = models.word2vec.Word2Vec(sentences,size = size,min_count=min_count,window = window)
		self.size = size
		#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level  =logging.INFO)

	def save(self,path=config.word2vec_path):
		self.model.save(path)
		print("word2vec model is saved at "+path)


	def load(self,path=config.word2vec_path):
		self.model = models.word2vec.Word2Vec.load(path)
		print("word2vec model is loaded at "+path)

	# 使えない。
	def __training(self,path):
		sentences = models.word2vec.Text8Corpus(path)
		#sentence = MySentences('/some/directory')
		self.model.train(sentences)

	# 使えない。
	def __trainingByStr(self,sentence):
		origins = self.tagger.parseOriginList(sentence)
		#print(origins)
		tags = sentence.split(" ")
		#print(tags)
		self.model.train(tags)

	# 使えない。
	def __trainingByDir(self,dirPath = config.parsedInputDirPath):
		paths = glob.glob(dirPath)
		print(paths)
		for path in paths:
			print("word2vec train at "+path)
			self.training(path)

	#文書を原型に直し、そのword2vecの和を文書ベクトルとする。
	def getSumVec(self,sentence):
		origins = self.tagger.parseOriginList(sentence)
		#origins[0]が入っていない場合、Keyerror発生
		#vec = np.array([0.0]*len(self.model[origins[0]]))
		vec = np.array([0.0]*200)
		for origin in origins:
			if origin in self.model.vocab.keys():
				vec += self.model[origin]
			else:
				#print(origin," is not included")
				pass
		return vec

	def getWord2Vec(self,word):
		if word in self.model.vocab.keys():
			return self.model[word]
		else :
			#print(word," is not included")
			#return [0 for i in range(self.size)]
			return np.array([0.0]*200)

	def getWord2VecByList(self,wordList):
		return np.array([self.getWord2Vec(word) for word in wordList])


##	ここがメインで呼び出されたら、デフォルト設定に対し、word2vec用にパース。
#	その後、word2vecを学習、保存
if __name__ == "__main__":
	model = word2vecManager(path = config.base+"text8")
	model.save()

