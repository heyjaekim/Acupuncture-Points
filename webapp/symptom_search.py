import re
from string import punctuation
from collections import defaultdict
import struct
from struct import pack, unpack
from os import listdir
from konlpy.tag import Okt
from math import sqrt
from math import log
import json

class Search_symptom:

    def __init__(self, query):
        self.pattern1 = re.compile(r'[{}]'.format(re.escape(punctuation)))
        self.pattern2 = re.compile(r'\b(\w|[.])+@(?:[.]?\w+)+\b')
        self.pattern3 = re.compile(r'\bhttps?://\w+(?:[.]?\w+)+\b')
        self.pattern4 = re.compile(r'[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]')
        self.pattern5 = re.compile(r'\b[a-z][A-Za-z0-9]+\b')
        self.pattern6 = re.compile(r'\s{2,}')
        self.pattern7 = re.compile(r'[a-z]')
        self.pattern8 = re.compile(r'[/,.]')
        self.okt = Okt()
        self.query = query

    def tokenizer1(self, doc): # 어절

        return doc.split()

    def tokenizer2(self, tokens, n=2): # 어절 Ngram

        ngram = list()
        for i in range(len(tokens) - (n-1)):
            ngram.append(tokens[i:i+n])
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

    def search(self):
        TF = lambda f, mf, a: a + (1 - a)*(f / mf)
        IDF = lambda N, df: log(N / df)

        with open('Files.json','r',encoding = 'utf-8') as f:
            FILES = json.load(f)
        N = len(FILES)

        wposting = 'wsymptom.dat'
        with open('globalDictionary.json','r', encoding = 'utf-8') as f:
            globalDictionary = json.load(f)

        queryDictionary = defaultdict(int)

        for _ in self.tokenizer1(self.query):
            queryDictionary[_] += 1
        for _ in self.tokenizer4(self.query):
            queryDictionary[_] += 1
        for _ in self.tokenizer5(self.query):
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
# a = Search_symptom('설사')
# a.search()[0][0], a.search()[2]
