from os import listdir, makedirs
from os.path import isfile, isdir, join
import cv2  
import copy
import json
import numpy as np

def open_temp_db():
    db = []
    dorsal_acupunctures = {'양계':'yanggye', '양지':'yangji', '외관':'oegwan', '양곡':'yanggok', 
                            '합곡':'hapgok', '중저':'jungjer', '삼간':'samgan', '이간':'egan',
                            '액문':'ekmoon','상양':'sangyang','중층':'jungcheung','소충':'sochung',
                            '소택':'sotack','관충':'gwanchung'}
    palmar_acupunctures = {'신문':'shinmoon','대릉':'daereung','태연':'taeyeon','어제':'urjae',
                            '소부':'sobu','노궁':'nogung','소상':'sosang'}

    db.append(dorsal_acupunctures)
    db.append(palmar_acupunctures)
    return db

def make_path_tuple(acupuncture_info, changed_hands_path):
    path = []
    for _ in changed_hands_path:
        for f in listdir(_):
            hand_pos = (_.split('/')[1]).strip(f'{acupuncture_info}_')
            path.append((join(_,f), hand_pos))
    return path

def is_acupuncture(ac, db):
    flag = False
    for _ in db:
        if ac in _:
            flag = True
            ac = _.get(ac)
            return ac
    if not flag:
        print("Can't find acupuncture info")
        exit()

def open_json_file(file_path):
    if isfile(file_path):
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    else:
        print("json file doesn't exist")
        exit()        

def save_json_file(json_data):
    with open(json_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)
          
def translate_image(images_info, json_data, xmove):

    save_directory = f'{acupuncture_info}_translated'
    if not(isdir(save_directory)):
        makedirs(save_directory)

    for i in range(0, len(images_info)):
        img_path = images_info[i][0]
        img_hand_pos = images_info[i][1]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_id = ((img_path.split('/')[-1]).split('_')[-1]).split('.')[0]
        rows,cols = img.shape[:2]

        # translate x-axis and y-axis of the image
        M = np.float32([[1,0,xmove],[0,1,0]])
        dst = cv2.warpAffine(img, M, (cols,rows))
 
        if xmove > 0:
            for row in range(len(dst)):
                for col in range(xmove):
                    if np.array_equal(dst[row,col], np.array([0,0,0])):
                        dst[row,col] = np.array([255,255,255])
        else:
            for row in range(len(dst)):
                for col in range(len(dst[row])-1,len(dst[row])+xmove-1,-1):
                    if np.array_equal(dst[row,col], np.array([0,0,0])):
                        dst[row,col] = np.array([255,255,255])
            

        acupuncture_id = f"{acupuncture_info}_{img_id}"
        x, y, xy = json_data[acupuncture_id][1].values()
        
        acupuncture_new_id = f"{acupuncture_info}_{img_id}" + f'+{xmove}' if xmove >= 0 else acupuncture_id + f'{xmove}'
        
        json_data[acupuncture_new_id] = list()
        json_data[acupuncture_new_id].append({
            "acup_info": f"{acupuncture_info}",
            "hand_pos": f"{img_hand_pos}",
            "acup_size": f"{acupuncture_size}"
        })
        json_data[acupuncture_new_id].append({
            "acup_coord_x": x + xmove,
            "acup_coord_y": y,
            "acup_coord": (x + xmove, y)
        })
        
        cv2.imwrite('./{0}/{1}.png'.format(save_directory, acupuncture_new_id), dst)
        

    save_json_file(json_data)

##############################################################################################

# 혈점 정보, 점 사이즈, 손 위치 입력
acupuncture_info = input('혈자리를 입력해주세요. ex) 소충 ')
xmove = int(input('이미지 x축 이동: ex) 10 or -10 '))

##############################################################################################

acupuncture_size = 3
acupuncture_db = open_temp_db()
acupuncture_info = is_acupuncture(acupuncture_info, acupuncture_db)

hand_path_frst = f'./{acupuncture_info}_dorsal_right/'
hand_path_scnd = f'./{acupuncture_info}_dorsal_left/'
hand_path_thrd = f'./{acupuncture_info}_palmar_right/'
hand_path_frth = f'./{acupuncture_info}_palmar_left/'

temp_paths = [hand_path_frst, hand_path_scnd, hand_path_thrd, hand_path_frth]
changed_hands_path = [p+'change' for p in temp_paths if isdir(p)]
orginal_hands_path = [p+'org' for p in temp_paths if isdir(p)]

json_file = "./{}_info.json".format(acupuncture_info)
json_data = open_json_file(json_file)

images_info = make_path_tuple(acupuncture_info, changed_hands_path)

translate_image(images_info, json_data, xmove)