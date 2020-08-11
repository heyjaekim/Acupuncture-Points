# TODO: 1. 형태소 분석기
# TODO: 2. 로직 설계 => 크롤링을 통해서 기대값에 상응하는 QnA json 파일을 가지고 sentence 안에 test context를 형태소 유사도 비교분석해서 평균적으로 몇퍼센트 나오는지 일단 구해보자  

test_context = ("저희 지역에서 코로나 바이러스 감염 확진자가 나왔습니다 제가 사는 동네 바로 옆동네이고요 저는 설날에는 다른 지역에 있다가 25일날\
     집으로 왔는데요 열이 나거나 기침이 나온다거나 그런 증상은 없는데 머리가 너무 깨질듯이 아픕니다 이건 코오나 바이러스 감염 아니겠죠",
            "머리가 깨질 듯이 아프고 눈도 무언가를 꾹 올려놨다가 뗀 것 처럼 아파요 이렇게 아픈건 처음인데 왜이러는건가요 학생이라 시험때문에 안먹던\
                커피를 하루에 한번씩 4일 정도 마시긴했어요 중간중간 새벽에 늦게 자기도 했고 혹시 카페인 때문일까요")

# both strings are the questions extracted from headache_info.json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity    
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from os.path import isfile
from json import load, dump
import numpy as np

def open_json_file(file_path):
    if isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_data = load(json_file)
        return json_data
    else:
        return dict()

def check_isfile(exp_ans):
    json_path = "./{}_info.json".format(exp_ans)
    while not isfile(json_path):
        exp_ans = input("expected answer: ")
        json_path = "./{}_info.json".format(exp_ans)
    return exp_ans, json_path

def check_similar_words(sim_set):
    for k,v in sim_set.items():
        if v == 1.0:
            print(k,v)

def l1_normalize(v):
    norm = np.sum(v)
    return v / norm

def check_similarities(sentence):
    # 객체 생성
    tfidf_vectorizer = TfidfVectorizer()
    # 문장 벡터화 진행
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
    # 각 단어
    text = tfidf_vectorizer.get_feature_names()
    # 각 단어의 벡터 값
    idf = tfidf_vectorizer.idf_

    sim_set = dict(zip(text, idf))

    # 동일한 단어
    check_similar_words(sim_set)
    # 코사인 유사도
    print("코사인 유사도: {}".format(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]))
    # 벡터 일반화
    tfidf_norm_l1 = l1_normalize(tfidf_matrix)
    print("--------------벡터 일반화--------------")
    # 유클리디언 유사도
    print("유클리디언 유사도: {}".format(euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])[0][0]))
    # 맨하탄 유사도
    print("벡터 일반화 맨하탄 유사도: {}".format(manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])[0][0]))


expected_ans = input("expected answer: ")
expected_ans, json_path = check_isfile(expected_ans)
json_data = open_json_file(json_path)

check_similarities(test_context)
