import numpy as np
import pandas as pd
import re

header_list = ['hand_num', 'hand_pose', 'yanggye_acup_name',
        'yanggye_acup_exist', 'yanggye_acup_x', 'yanggye_acup_y',
        'yangji_acup_name', 'yangji_acup_exist', 'yangji_acup_x',
        'yangji_acup_y', 'oegwan_acup_name', 'oegwan_acup_exist',
        'oegwan_acup_x', 'oegwan_acup_y', 'yanggok_acup_name',
        'yanggok_acup_exist', 'yanggok_acup_x', 'yanggok_acup_y',
        'hapgok_acup_name', 'hapgok_acup_exist', 'hapgok_acup_x',
        'hapgok_acup_y', 'jungjer_acup_name', 'jungjer_acup_exist',
        'jungjer_acup_x', 'jungjer_acup_y', 'samgan_acup_name',
        'samgan_acup_exist', 'samgan_acup_x', 'samgan_acup_y',
        'egan_acup_name', 'egan_acup_exist', 'egan_acup_x',
        'egan_acup_y', 'ekmoon_acup_name', 'ekmoon_acup_exist',
        'ekmoon_acup_x', 'ekmoon_acup_y', 'sangyang_acup_name',
        'sangyang_acup_exist', 'sangyang_acup_x', 'sangyang_acup_y',
        'jungcheung_acup_name', 'jungcheung_acup_exist',
        'jungcheung_acup_x', 'jungcheung_acup_y', 'sochung_acup_name',
        'sochung_acup_exist', 'sochung_acup_x', 'sochung_acup_y',
        'sotack_acup_name', 'sotack_acup_exist', 'sotack_acup_x',
        'sotack_acup_y', 'gwanchung_acup_name', 'gwanchung_acup_exist',
        'gwanchung_acup_x', 'gwanchung_acup_y', 'shinmoon_acup_name',
        'shinmoon_acup_exist', 'shinmoon_acup_x', 'shinmoon_acup_y',
        'daereung_acup_name', 'daereung_acup_exist', 'daereung_acup_x',
        'daereung_acup_y', 'taeyeon_acup_name', 'taeyeon_acup_exist',
        'taeyeon_acup_x', 'taeyeon_acup_y', 'urjae_acup_name',
        'urjae_acup_exist', 'urjae_acup_x', 'urjae_acup_y',
        'sobu_acup_name', 'sobu_acup_exist', 'sobu_acup_x',
        'sobu_acup_y', 'nogung_acup_name', 'nogung_acup_exist',
        'nogung_acup_x', 'nogung_acup_y', 'sosang_acup_name',
        'sosang_acup_exist', 'sosang_acup_x', 'sosang_acup_y']
        
data1 = pd.DataFrame(None, columns=header_list)

def return_only_number_in_list(list_name):
    new_list_name = []
    for i in range(0,len(list_name)):
        k = re.split('_',list_name[i])[1].split('.')[0]
        new_list_name.append(k)
    return(new_list_name)

data1['hand_num'] = return_only_number_in_list(pd.read_csv('HandInfo.csv').imageName)
data1['hand_pose'] = pd.read_csv('HandInfo.csv').aspectOfHand

import json
from os.path import isfile

def open_json_file(file_name):
    if isfile(file_name):
        with open(file_name, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    else:
        return dict()

input_ = input('json file name : ')
a_json=open_json_file(input_)
# a_json=open_json_file('gwanchung_info.json')
# input_ must be 'cleaned json file' (rf. preprocessing_json.py)

for i,j in enumerate(return_only_number_in_list(list(a_json.keys()))):
    if j in list(data1.hand_num):
        acup_name_i = list(a_json.values())[i][0]['acup_info']
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_exist']=1
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_x'] = list(a_json.values())[i][1]['acup_coord_x']
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_y'] = list(a_json.values())[i][1]['acup_coord_y']
    else:
        data1 = data1.append(pd.Series(np.repeat(None,86)), ignore_index=True)
        data1.iloc[-1,:].hand_num = j
        data1.loc[data1.hand_num == j,'hand_pose'] = list(a_json.values())[i][0]['hand_pos']
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_exist']=1
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_x'] = list(a_json.values())[i][1]['acup_coord_x']
        data1.loc[data1.hand_num == j,acup_name_i+'_acup_y'] = list(a_json.values())[i][1]['acup_coord_y']

###########################
data1.yanggye_acup_name = 'yanggye'
data1.yangji_acup_name = 'yangji'
data1.oegwan_acup_name = 'oegwan'
data1.yanggok_acup_name = 'yanggok'
data1.hapgok_acup_name = 'hapgok'
data1.jungjer_acup_name = 'jungjer'
data1.samgan_acup_name = 'samgan'
data1.egan_acup_name = 'egan'
data1.ekmoon_acup_name = 'ekmoon'
data1.sangyang_acup_name = 'sangyang'
data1.jungcheung_acup_name = 'jungcheung'
data1.sochung_acup_name = 'sochung'
data1.sotack_acup_name = 'sotack'
data1.gwanchung_acup_name = 'gwanchung'
data1.sobu_acup_name = 'sobu'
data1.nogung_acup_name = 'nogung'
data1.urjae_acup_name = 'urjae'
data1.shinmoon_acup_name = 'shinmoon'
data1.hapgok_acup_name = 'hapgok'
data1.daereung_acup_name = ' daereung'
data1.sosang_acup_name = 'sosang'

for i in range(len(data1)):
    if (data1.hand_pose[i] == 'palmar right') | (data1.hand_pose[i] == 'palmar left'):
        data1.yanggye_acup_exist[i] = 0
        data1.yangji_acup_exist[i] = 0
        data1.oegwan_acup_exist[i] = 0
        data1.yanggok_acup_exist[i] = 0
        data1.hapgok_acup_exist[i] = 0
        data1.jungjer_acup_exist[i] = 0
        data1.samgan_acup_exist[i] = 0
        data1.egan_acup_exist[i] = 0
        data1.ekmoon_acup_exist[i] = 0
        data1.sangyang_acup_exist[i] = 0
        data1.jungcheung_acup_exist[i] = 0
        data1.sochung_acup_exist[i] = 0
        data1.sotack_acup_exist[i] = 0
        data1.gwanchung_acup_exist[i] = 0
    
    elif (data1.hand_pose[i] == 'dorsal right') | (data1.hand_pose[i] == 'dorsal left'):
        data1.sobu_acup_exist[i] = 0
        data1.nogung_acup_exist[i] = 0
        data1.urjae_acup_exist[i] = 0
        data1.shinmoon_acup_exist[i] = 0
        data1.hapgok_acup_exist[i] = 0
        data1.daereung_acup_exist[i] = 0
        data1.sosang_acup_exist[i] = 0
        
output_ = input('output csv file name : ')
data1.to_csv('./'+output_,header=True,index=False)
