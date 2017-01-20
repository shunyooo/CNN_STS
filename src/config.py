## 	@package 大局変数管理用。pathなど。
#

base = "../STS_data/"

inputDirPath= base+'input/*'	#学習用データ群
gsDirPath	= base+'gs/*'		#学習回答データ群

linedInputDirPath= base+'parsed/lined/input/*' #学習用、テスト用データを文書毎に改行しまとめたデータ群
parsedInputDirPath = base+'parsed/origined/input/*'#学習、テスト用データを原型に整形しまとめたもの 。改行区切りでword2vec学習用。

parsedAllPath = base+'parsed/origined/all/all.txt'

router_origined_base = base + "parsed/origined/router/"
router_all_key_path = 'temp/reuter_all.keys'

targetDirPath = base+'target/14STS.input.images.txt'#テスト対象データ
outPutFilePath = base+"output/result.txt"
ansCosFilePath = base+"ans/14STS.gs.images.txt"

## treetaggerファイルのパス
treetagger_path = 'tree-tagger'

## word2vecに関するパス
word2vec_path 	= 'temp/word2vec.model'

## core nlpファイルPath
corenlp_path = "../stanford-corenlp-full-2014-08-27"
properties_file = "../user.properties"