import json
import os
from os.path import isfile, isdir, join
import re

# 현재 디렉토리에 있는 파일들 중 확장자가 .json인 파일만을 출력.
def print_json_filename(dirname):
    full_filenames = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join('', filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.json': 
            full_filenames.append(full_filename)
    print('json파일이',len(full_filenames),'개 있습니다.')
    return(full_filenames)
    
# json 파일 열기
def open_json_file(file_name):
    if isfile(file_name):
        with open(file_name, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    else:
        return dict()

# json 파일 cleaning
def cleaning_json(file_name):
    json_file = open_json_file(file_name)
    for i in list(json_file.keys()):
        try:
            json_file[i][1]
        except IndexError:
            json_file.pop(i)
    print('cleaned')

# json merge(update)
def json_merge(jf1, jf2):
    return(jf1.update(jf2))
    
# dir 안의 png파일 이름 모두 불러오기 -> list 형태로 저장
def search(dirname):
    filenames = os.listdir(dirname)
    filename_list = []
    for filename in filenames:
        full_filename = os.path.join('', filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.png': 
            filename_list.append(full_filename)
    return(filename_list)

# 숫자만 리스트 형태로 저장
def return_only_number_in_list(list_name):
    new_list_name = []
    for i in range(0,len(list_name)):
        k = re.split('_',list_name[i])[1].split('.')[0]
        new_list_name.append(k)
    return(new_list_name)

def checking_symmetric_difference(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if (set1-set2==set())&(set2-set1==set()):
        print('교집합만 존재함')
    else:
        print('차집합이 존재함')
        print('first input에만 있는 것:',set1-set2)
        print('second input에만 있는 것:',set2-set1)
        
# 교집합 만들기
def make_intersection(f1, f2, f3):
    set1 = set(f1)
    set2 = set(f2)
    set3 = set(f3)
    intersection_set = set1 & set2 & set3

# json 파일 생성
def save_json_file(json_data):
    with open(json_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)
        
# 통합.. (구현중)
def json_intersection_extract(dirname):
  pass
