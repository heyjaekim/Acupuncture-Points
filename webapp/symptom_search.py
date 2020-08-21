import re
from string import punctuation
from collections import defaultdict
from struct import pack, unpack
from os import listdir
from konlpy.tag import Okt
from math import sqrt
from math import log

# 증상 검색
class Search_symptom:

    def __init__(self, query):
        self.pattern1 = re.compile(r'[{}]'.format(re.escape(punctuation)))
        self.pattern2 = re.compile(r'\b(\w|[.])+@(?:[.]?\w+)+\b')
        self.pattern3 = re.compile(r'\bhttps?://\w+(?:[.]?\w+)+\b')
        self.pattern4 = re.compile(r'[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]')
        self.pattern5 = re.compile(r'\b[a-z][A-Za-z0-9]+\b')
        self.pattern6 = re.compile(r'\s{2,}')

        self.okt = Okt()
        self.query = query

    # 증상 정렬 파일 탐색
    def fileids(self, path = './symp/'):

        return [path+_ for _ in listdir(path) if re.search('[.]txt$', _)]

    def cleaning(self, doc): # 전처리

        return self.pattern6.sub(' ',
               self.pattern1.sub(' ',
               self.pattern5.sub(' ',
               self.pattern4.sub(' ',
               self.pattern2.sub(' ', doc))))).strip()

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

    def get_tokens(self, file): # 단어를 작은 단위로 분리

        terms = defaultdict(lambda:0)

        with open(file, 'r', encoding='cp949') as f:
            self.news = self.cleaning(f.read())

        for _ in self.tokenizer1(self.news):

            terms[_] += 1

        for _ in self.tokenizer4(self.news):

            terms[_] += 1
        for _ in self.tokenizer5(self.news):

            terms[_] += 1

        return terms

    def indexer(self, file): # 각 어구, 어절, N-gram 마다 인덱싱

        lexicon = defaultdict(int)
        for k, v in self.get_tokens(file).items():
            lexicon[k] += v

        return lexicon

    def mergeLexicon(self, Lexicon, i, LocalLexicon, posting):

        with open(posting, 'ab') as f:
            for k, v in LocalLexicon.items():
                termInfo = (i, v, Lexicon[k] if k in Lexicon.keys() else -1)
                Lexicon[k] = f.tell()
                f.write(pack('iii', termInfo[0], termInfo[1], termInfo[2]))

        return Lexicon

    def sortedLexicon(self, Lexicon, posting, sposting): # 역 색인 구조

        sortedIndex = list()

        f1 = open(posting, 'rb')
        f2 = open(sposting, 'wb')

        for k, v in Lexicon.items():
            pos1 = v
            pos2 = f2.tell()
            df = 0

            while pos1 > -1:
                f1.seek(pos1)
                termInfo = unpack('iii', f1.read(12))
                f2.write(pack('ii', termInfo[0], termInfo[1]))
                df += 1
                pos1 = termInfo[-1]

            Lexicon[k] = (df, pos2)
        f1.close()
        f2.close()
        return Lexicon

    # TF-IDF를 이용하여 가장 적합한 문서 검색
    def search_engine(self):

        TF = lambda f, mf, a: a + (1 - a)*(f / mf)
        IDF = lambda N, df: log(N / df)

        FILES = [{'path':_, 'maxfreq':0, 'length':0} for _ in self.fileids()]

        N = len(FILES)

        posting = 'symptom.dat'
        sposting = 'ssymptom.dat'
        wposting = 'wsymptom.dat'

        globalDictionary = dict()

        for docInfo in FILES:
            i = FILES.index(docInfo)
            localDictionary = self.indexer(docInfo['path'])
            FILES[i]['maxfreq'] = max(localDictionary.values())
            globalDictionary = self.mergeLexicon(globalDictionary, i, localDictionary, posting)

        globalDictionary = self.sortedLexicon(globalDictionary, posting, sposting)

        f1 = open(sposting, 'rb')
        f2 = open(wposting, 'wb')

        for k, v in globalDictionary.items():
            docfreq, pos1 = v
            f1.seek(pos1)
            pos2 = f2.tell()

            for _ in range(docfreq):
                docid, termfreq = unpack('ii', f1.read(8))
                w = TF(termfreq, FILES[docid]['maxfreq'], 0) * IDF(N, docfreq)
                f2.write(pack('if', docid, w))
                FILES[docid]['length'] += w ** 2

            globalDictionary[k] = (docfreq, pos2)

        f1.close()
        f2.close()


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
                return _, f.read().strip()

#############################################################################
# a = Search_symptom('감기')
# result = a.search_engine()[0][0]
# print(result)