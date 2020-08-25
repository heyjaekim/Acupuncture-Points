import re
from string import punctuation
from collections import defaultdict
import struct
from struct import pack, unpack
from os import listdir
<<<<<<< HEAD:webapp/Text_Searching/Symptom_Search/search_symptom.py
from konlpy.tag import Okt, Kkma
=======
from konlpy.tag import Okt
from konlpy.tag import Kkma
>>>>>>> 5c344fb6dfee5149938d1c51e8f1262f35a54c83:Text Searching/Symptom_Search/Search ver.0825/search_symptom.py
from math import sqrt
from math import log
import json

class Search_symptom:

    def __init__(self):
        self.pattern1 = re.compile(r'[{}]'.format(re.escape(punctuation)))
        self.pattern2 = re.compile(r'\b(\w|[.])+@(?:[.]?\w+)+\b')
        self.pattern3 = re.compile(r'\bhttps?://\w+(?:[.]?\w+)+\b')
        self.pattern4 = re.compile(r'[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]')
        self.pattern5 = re.compile(r'\b[a-z][A-Za-z0-9]+\b')
        self.pattern6 = re.compile(r'\s{2,}')
        self.pattern7 = re.compile(r'[a-z]')
        self.pattern8 = re.compile(r'[/,.]')
        self.okt = Okt()
        self.kkma = Kkma()

    def spacing_kkma(self, wrongSentence):
        pattern = re.compile('[ㄱ-ㅎㅏ-ㅣ]')
        tagged = self.kkma.pos(wrongSentence)
        corrected = ""
        for i in tagged:
            if i[1][0] in "JEXSO":
                corrected += i[0]
            else:
<<<<<<< HEAD:webapp/Text_Searching/Symptom_Search/search_symptom.py
                corrected += " " + i[0]
=======
                corrected += " "+i[0]
>>>>>>> 5c344fb6dfee5149938d1c51e8f1262f35a54c83:Text Searching/Symptom_Search/Search ver.0825/search_symptom.py
        if corrected[0] == " ":
            corrected = corrected[1:]

        res = pattern.sub('', corrected)
        return res

    def tokenizer1(self, doc): # 어절

        return doc.split()

    def tokenizer2(self, doc, n=2): # 어절 Ngram
        tmp = doc.split()
        ngram = list()
        for i in range(0, len(tmp) - n + 1):
            token = ''
            for j in range(i, i + n):
                token += tmp[j] + ' '
            ngram.append(token)
        return ngram

    def tokenizer3(self, doc, n=2): # 음절 Ngram

        ngram = list()
        for i in range(len(doc) - (n-1)):
            ngram.append(doc[i:i+n])
        return ngram

    def tokenizer4(self, doc): # 형태소

        return [_ for _ in self.okt.morphs(doc) if 1 < len(_) < 8]

    def tokenizer5(self, doc): # 명사

        return [_ for _ in self.okt.nouns(doc) if 1 < len(_) < 8]

    def fileids(self, path = './symp/'):
        return [path+_ for _ in listdir(path) if re.search('[.]txt$', _)]


    def search(self, query):
        TF = lambda f, mf, a: a + (1 - a)*(f / mf)
        IDF = lambda N, df: log(N / df)

        with open('./Text_Searching/Symptom_Search/FILES.json', 'r', encoding ='utf-8') as f:
            FILES = json.load(f)
        N = len(FILES)

        wposting = './Text_Searching/Symptom_Search/wsymptom.dat'
        with open('./Text_Searching/Symptom_Search/globalDictionary.json', 'r', encoding ='utf-8') as f:
            globalDictionary = json.load(f)

        queryDictionary = defaultdict(int)

        for _ in self.tokenizer1(query):
            queryDictionary[_] += 1
        for _ in self.tokenizer4(query):
            queryDictionary[_] += 1
        for _ in self.tokenizer5(query):
            queryDictionary[_] += 1

        qmaxfreq = max(queryDictionary.values())
        querylength = 0
        queryWeight = defaultdict(int)
        for k, v in queryDictionary.items():
            if k in globalDictionary:
                queryWeight[k] = TF(v, qmaxfreq, 0.5) * IDF(N, globalDictionary[k][0])
                querylength += queryWeight[k]**2


        result = defaultdict(int)

        with open(wposting, 'rb') as f:
            for k, v in queryWeight.items():
                df, pos = globalDictionary[k]
                f.seek(pos)
                for _ in range(df):
                    docid, weight = unpack('if', f.read(8))
                    result[docid] += weight*v

        for k, v in result.items():
            result[k] = v / (sqrt(FILES[k]['length']) * sqrt(querylength))


        K = 1
        for _ in list(sorted(result.items(), key=lambda _:-_[1]))[:K]:

            with open(FILES[_[0]]['path'], 'r') as f:
                return _, f.read().strip(), self.pattern7.sub('',self.pattern8.sub('', FILES[_[0]]['path']))

######################################################################################################
<<<<<<< HEAD:webapp/Text_Searching/Symptom_Search/search_symptom.py
# query = '머리아프고, 속이 울렁거린다'
#
#
# a = Search_symptom()
# b = tokenizer2(query)
# print(a.search_engine(b[0]))
# print(a.search_engine(b[1]))
=======
query = '머리아프고, 속이 울렁거린다'


a = Search_symptom()
query_change = a.spacing_kkma(query)
b = a.tokenizer2(query_change)
print(a.search(b[0]))
print(a.search(b[1]))
>>>>>>> 5c344fb6dfee5149938d1c51e8f1262f35a54c83:Text Searching/Symptom_Search/Search ver.0825/search_symptom.py
