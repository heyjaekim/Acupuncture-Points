import json
import os
from os.path import isfile, join
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
def cleaning_json(json_file):
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

# 통합
def json_intersection_extract(dirname,acup_left_dirname, acup_right_dirname,acup_name):
    json_filename_list = print_json_filename(dirname)
    print(json_filename_list)
    
    json_to_dictionary = {}
    for i in range(len(json_filename_list)):
        json_to_dictionary.update(open_json_file(json_filename_list[i]))
    
    
    cleaning_json(json_to_dictionary)
    
    json_file_keys = return_only_number_in_list(list(json_to_dictionary.keys()))
    
    left_intersection = set(return_only_number_in_list(search(join(acup_left_dirname,'change')))) & set(return_only_number_in_list(search(join(acup_left_dirname,'org'))))
    right_intersection = set(return_only_number_in_list(search(join(acup_right_dirname,'change')))) & set(return_only_number_in_list(search(join(acup_right_dirname,'org'))))

    lr_joined_set = left_intersection | right_intersection
    
    
    completed_set = set(json_file_keys) & lr_joined_set
    
    c = set()
    for i in completed_set:
        k = acup_name+'_'+i
        c.add(k)

    d = dict()
    for j in c:
        d[j]=json_to_dictionary[j]
    
    with open(acup_name + '_intersection.json', 'w') as make_file:
        json.dump(d, make_file, indent="\t")
        
# how to use : json_intersection_extract('./','./sangyang_dorsal_left','./sangyang_dorsal_right','sangyang')
