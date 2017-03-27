## 	@package 構文解析管理用パッケージ
#	基本的に原型に直す用
#	構文解析用ファイルtree-taggerとtreetaggerwrapperが同ディレクトリ内に必要。
import treetaggerwrapper
import re
import config
import sys
import time
import timeout_decorator


##	構文解析用
class TRTG_NLP():
	def __init__(self):
		self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR=config.treetagger_path)
	
	@timeout_decorator.timeout(2)
	def tag_text(self,sentence):
		return self.tagger.tag_text(sentence)

	##	与えられた文書を単語毎に「単語,形態素,原型」の形に直す
	#	文書[[[単語][形態素][原型]],....,]
	#	記号の削除,全てを小文字への変換
	#	@param sentence は対象となる文書。
	def tagging(self,sentence):
		#print(sentence)
		tagedSentence = []
		#記号の除去
		sentence = re.sub(r'[^a-z| ]'," ",sentence.lower())
		try:
			tags = self.tag_text(sentence)
		except:
			print("taggerでエラー")
			print("文章:\n"+sentence)
			print("エラー内容\n",sys.exc_info())
			return []

		for tag in tags:
			#print(tag)
			tag = tag.split("	")
			tagedSentence.append(tag)
		return tagedSentence

	##	原型のリストに変換(順番準拠)
	#	@param sentence は対象となる文書。
	#	変換できなかった時は、空のリストを返すので、対応した処理を呼び出し元でおこなってください。
	def parseOriginList(self,sentence):
		origins = []
		for tag in self.tagging(sentence):
			#print(tag)	
			if len(tag) == 3:
				origins.append(tag[2])
		return origins

	#	原型の文(String)に変換、返却する。つまり原型の分かち書き文書を返す。
	#	@param sentence は対象となる文書。
	def parseOrigin(self,sentence):
		return " ".join(self.parseOriginList(sentence))