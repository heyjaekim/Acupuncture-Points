from requests import request
from requests.compat import urljoin, urlparse
from requests.exceptions import HTTPError
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from time import sleep
from string import punctuation
from json import load, dump
from os.path import isfile

import re
pattern1 = re.compile('[{}]'.format(re.escape(punctuation)))
pattern2 = re.compile(r'\b(\w|[.])+@(?:[.]?\w+)+\b')
pattern3 = re.compile(r'\bhttps?://\+(?:[.]?\w+)+\b')
pattern4 = re.compile(r'\b[^A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ ]')
pattern5 = re.compile(r'\b[a-z][A-Za-z0-9]+\b')
pattern6 = re.compile(r'\s{2,}')
pattern7 = re.compile(r'\b[A-Za-z0-9 ]+\b')

headers = {
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                    AppleWebKit/537.36 (KHTML, like Gecko)\
                        Chrome/84.0.4147.105 Safari/537.36'
}

def open_json_file(file_path):
    if isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_data = load(json_file)
        return json_data
    else:
        return dict()        

def save_json_file(json_data):
    with open(json_file, 'w', encoding='utf-8') as make_file:
        dump(json_data, make_file, ensure_ascii=False, indent="\t")


def cleaning(doc):
    return pattern7.sub(' ',
            pattern6.sub(' ', 
            pattern1.sub(' ',
            pattern5.sub(' ',
            pattern4.sub(' ',
            pattern2.sub(' ', doc)))))).strip()

def canfetch(url, agent='*', path='/'):
    robot = RobotFileParser(urljoin(url, '/robots.txt'))
    robot.read()
    return robot.can_fetch(agent, urlparse(url)[2])
    
def download(url, params={}, headers={}, method='GET', limit=3):
    if canfetch(url) == False:
        print('[Error] ' + url)
    try:
        resp = request(method, url,
               params=params if method=='GET' else {},
               data=params if method=='POST' else {},
               headers=headers)
        resp.raise_for_status()
    except HTTPError as e:
        if limit > 0 and e.response.status_code >= 500:
            print(limit)
            time.sleep(1) # => random
            resp = download(url, params, headers, method, limit-1)
        else:
            print('[{}] '.format(e.response.status_code) + url)
            print(e.response.status_code)
            print(e.response.reason)
            print(e.response.headers)
    return resp

def kin_crawling(url, kin_data):
    urls, visited = list(), list()
    ques_n = 0

    urls.append({'url':url, 'depth':1})

    while urls and ques_n <= 10: # Queue
        seed = urls.pop(0) # BFS
        visited.append(seed)
        if seed['depth'] > 15:
            break
        
        resp = download(seed['url'])
        dom = BeautifulSoup(resp.text, 'html.parser')
        
        # 크롤링
        for _ in dom.select('ul.basic1 dt > a[href], div.s_paging > a[href]'):
            newurl = _['href']
            if not 'https' in newurl:
                newurl = urljoin("https://kin.naver.com", newurl)

            if newurl not in [_['url'] for _ in urls] and\
                newurl not in [_ for _ in visited]:
                visited.append(newurl)
                urls.append({'url':newurl,
                            'depth':seed['depth']+1})
 
        # 스크래핑
        try:
            if dom.select_one('div.c-heading__content') and len(dom.select('div.se-component-content')) > 0 and "전문의" in dom.select_one('div.c-heading-answer__title > div.c-userinfo span:nth-of-type(2)'):
                    ques = cleaning(dom.select_one('div.c-heading__content').text)
                    ans = cleaning(str(dom.select('div.se-component-content')))
                    print("success1")
                    ques_key = "Kin_"+str(ques_n)
                    kin_data[ques_key] = {
                        "question":f"{ques}",
                        "answer":f"{ans}"}
                    ques_n += 1
            
            elif dom.select_one('div.c-heading__content') and len(dom.select('div.c-heading-answer__content-user p')) > 0 and "전문의" in dom.select_one('div.c-heading-answer__title > div.c-userinfo span:nth-of-type(2)'):
                    ques = cleaning(dom.select_one('div.c-heading__content').text)
                    ans = cleaning(str(dom.select('div.c-heading-answer__content-user p')))
                    print("success2")
                    ques_key = "Kin_"+str(ques_n)
                    kin_data[ques_key] = {
                        "question":f"{ques}",
                        "answer":f"{ans}"}
                    ques_n += 1
        except TypeError:
            pass
    save_json_file(kin_data)

#######################################################################################

search = input('Naver Kin Search Word: ')
expected_ans = input('Type in expected answer: ')

# search = ['다한증', '두근거림', '두통', '발열', '백내장', '복통', '부정맥', '불면', '코피', '산후무유', '수유량', '생리통', '설사', '소화장애', '손목 아픔', '손바닥 열', '수관절장', '수족냉증', '실신', '안구피로']
# expected_ans = ['다한증', '두근거림', '두통', '발열', '백내장', '복통', '부정맥', '불면', '코피', '산후무유', '산후무유', '생리통', '설사', '소화장애', '무력', '손바닥', '수관절장', '수족냉증', '실신', '안구피로']
# search = ['열사병','오심','요로폐색','요실금','요통',
#         '이명','인후염','입덧','족관절염좌','주관절통증',
#         '중풍','천식','치통','코감기','편도선염','피로',
#         '항강','현훈','황달','흉통','히스테리']
# expected_ans = ['열사병','오심','요로폐색','요실금','요통',
#         '이명','인후염','입덧','족관절염좌','주관절통증',
#         '중풍','천식','치통','코감기','편도선염','피로',
#         '항강','현훈','황달','흉통','히스테리']
#######################################################################################
try:
    search_url = search.replace(' ', '+')
    url = 'https://kin.naver.com/search/list.nhn?query=/'+search_url
    params = {'query':''}
    params['query'] = search
    json_file = "./Crawling_data/scrapped-data/{0}_{1}_info.json".format(search, expected_ans)
    json_data = open_json_file(json_file)
    kin_crawling(url, json_data)
except ConnectionError:
    pass

# for i in range(len(search)):
#     try:
#         search_url = search[i].replace(' ', '+')
#         url = 'https://kin.naver.com/search/list.nhn?query=/'+search_url
#         params = {'query':''}
#         params['query'] = search[i]
#         json_file = "./Crawling_data/scrapped-data/{0}_{1}_info.json".format(search[i], expected_ans[i])
#         json_data = open_json_file(json_file)
#         kin_crawling(url, json_data)
#     except ConnectionError:
#         pass