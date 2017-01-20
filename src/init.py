import pickle
# 構文解析
from corenlpManager import CORE_NLP
from treetaggerManager import TRTG_NLP
from word2vecManager import word2vecManager
from progressbar import ProgressBar
import time

import time

import config
import glob
import os
import sys
import locale

import codecs

import pprint

#データのパースや、pickle処理
if __name__ == "__main__":

	print("word2vecの学習")
	model = word2vecManager(path = config.base+"text8")
	model.save()

	print("文書データのパース")
	#パースしてファイルを書き直し。
	#input/*から
	#parsed/origined/input/*へ

	paths = glob.glob(config.inputDirPath)
	parser = CORE_NLP()

	writePath = config.parsedInputDirPath
	writePath, filePath = os.path.split(writePath)
	writePath += "/"

	for i,path in enumerate(paths):
		dirPath, filePath = os.path.split(path)
		outputPath = writePath+filePath
		print("path: read-->",path," output-->",outputPath)
		with open(path,'r') as f,open(outputPath,'w') as output:
			for line in f:
				pair_of_doc = line[:-1].split('\t')
				#print(pair_of_doc,i)
				output.write(parser.parseOrigin(pair_of_doc[0])+"\t")
				output.write(parser.parseOrigin(pair_of_doc[1])+"\n")









