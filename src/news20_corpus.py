import pickle
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
pp = pprint.PrettyPrinter(indent=4)


#news20に関して、各文書を原型にパース、タグづけを行う。
#corenlpでは時間がかかるので、treetaggerを採用。
#1文書は以下
#{	
#	"tag":"タグ名",
#	"no":tag_number,
#	"doc":"文書",
#	"origin_doc":{"word","word",..},
#	"word2vec_list":{"word2vec","word2vec",...}
#}
if __name__ == "__main__":
	#文書の単語数。ここで切る。
	DOCUMENT_LENGTH = 1000
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


	parser = TRTG_NLP()
	word2vecer = word2vecManager()
	word2vecer.load()
	print(len(word2vecer.model.vocab.keys()))

	path = config.base+"20news/*"
	#ファイルの読み込み
	category_paths = glob.glob(path)
	#print(category_paths)

	for category_path in category_paths:
		data = []
		_, category = os.path.split(category_path)
		category_path += "/*"
		doc_paths = glob.glob(category_path)

		#if category in category_list[:2]:
		#	continue

		#１文書ごとにfor文
		#	- 原型への変換
		#	- word2vecへの変換
		print("category -> "+category)
		print("doc num  -> "+str(len(doc_paths)))
		p = ProgressBar(maxval=len(doc_paths))
		for i,doc_path in enumerate(doc_paths):
			f = codecs.open(doc_path, 'r', 'utf8', 'ignore')
			doc = f.read()

			doc_feature = {}
			doc_feature["tag"] = category
			doc_feature["no"] = category_list.index(category)
			#doc_feature["doc"] = doc
			doc_feature["origin_doc"] = parser.parseOriginList(doc)
			if len(doc_feature["origin_doc"]) == 0:
				continue

			# DOCUMENT_LENGTHで固定。
			if len(doc_feature["origin_doc"]) > DOCUMENT_LENGTH:
				doc_feature["origin_doc"] = doc_feature["origin_doc"][:DOCUMENT_LENGTH]
			elif len(doc_feature["origin_doc"]) < DOCUMENT_LENGTH:
				doc_feature["origin_doc"].extend(["thisisanull"]*(DOCUMENT_LENGTH-len(doc_feature["origin_doc"])))
			#print(len(doc_feature["origin_doc"]))

			doc_feature["word2vec_list"] = word2vecer.getWord2VecByList(doc_feature["origin_doc"])

			#print(doc_feature["doc"])
			#print(doc_feature["origin_doc"])
			#print(doc_feature["word2vec_list"])
			#sprint(len(doc_feature["origin_doc"]))

			data.append(doc_feature)
			p.update(i+1)

		with open('temp/20news/'+category,"wb") as output:
			pickle.dump(data,output)
			print('temp/20news/'+category+" に保存。")




