import pprint
import json
import corenlp
import config
import glob
import os
import re


##	構文解析用
class CORE_NLP():
	def __init__(self):
		self.parser = corenlp.StanfordCoreNLP(corenlp_path=config.corenlp_path,properties=config.properties_file)

	##	原型のリストに変換(順番準拠)
	#	@param sentence は対象となる文書。
	def parseOriginList(self,sentence):
		origins = []
		# 記号を削除
		print("元文書　　　　　:",sentence)
		sentence = re.sub(r'[^a-z| |\']'," ",sentence.lower())
		print("記号、小文字揃え:",sentence)
		result_json = json.loads(self.parser.parse(sentence))
		for word in result_json['sentences'][0]['words']:
			origins.append(word[1]['Lemma'])
		print("原型パース　　　:"," ".join(origins))
		print()
		return origins

	#	原型の文(String)に変換、返却する。つまり原型の分かち書き文書を返す。
	#	@param sentence は対象となる文書。
	def parseOrigin(self,sentence):
		return " ".join(self.parseOriginList(sentence))


if __name__ == "__main__":
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


	

