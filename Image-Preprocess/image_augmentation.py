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
            hand_pos = (_.split('/')[2]).strip(f'{acupuncture_info}_')
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
        json.dump(json_data, outfile, indent='\t')
          
def translate_image(images_info, json_data, xmove):

    save_directory = f'./{acupuncture_info}/_translated'
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
        
        acupuncture_new_id = acupuncture_id + f'_tr{xmove}' if xmove >= 0 else acupuncture_id + f'_tr{xmove}'
        
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

def scaling_image(images_info, json_data, dim):

    save_directory = f'./{acupuncture_info}/_scaled'
    if not(isdir(save_directory)):
        makedirs(save_directory)
    
    x_offset = [0, 40, 80, 120, 160]
    # x_offset = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    for i in range(0, len(images_info)):
        img_path = images_info[i][0]
        img_hand_pos = images_info[i][1]
        # 왼쪽 위로 줄어들게 된다: 700 - dim = xmove
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), dsize=(dim,dim))
        img_id = ((img_path.split('/')[-1]).split('_')[-1]).split('.')[0]
        height, width = img.shape[:2]
        
        blank_image = np.zeros((700,700,3), np.uint8)
        blank_image[:,:] = (255,255,255)
        l_img = blank_image.copy()
        for xmove in x_offset:
            if dim + xmove < 700:
                for row in range(len(img)):
                    for col in range(len(img)):
                        l_img[row, col+xmove] = img[row, col]
                        
                acupuncture_id = f"{acupuncture_info}_{img_id}"
                acupuncture_new_id = acupuncture_id + f'_dm{dim}_sc{xmove}'
                x, y, xy = json_data[acupuncture_id][1].values()
                new_x, new_y = x * dim / 700 + xmove, y * dim / 700
                new_xy = (new_x, new_y)

                json_data[acupuncture_new_id] = list()
                json_data[acupuncture_new_id].append({
                    "acup_info": f"{acupuncture_info}",
                    "hand_pos": f"{img_hand_pos}",
                    "acup_size": f"{acupuncture_size}"
                })
                json_data[acupuncture_new_id].append({
                    "acup_coord_x": new_x,
                    "acup_coord_y": new_y,
                    "acup_coord": new_xy
                })

                cv2.imwrite(f'./{save_directory}/{acupuncture_new_id}.png', l_img)
                l_img = blank_image.copy()
    save_json_file(json_data)

def rotate_image(images_info, json_data, angle):

    save_directory = f'./{acupuncture_info}/_scaled'
    if not(isdir(save_directory)):
        makedirs(save_directory)
    
    x_offset = [0, 40, 80, 120, 160]
    # x_offset = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    for i in range(0, len(images_info)):
        img_path = images_info[i][0]
        img_hand_pos = images_info[i][1]
        # 왼쪽 위로 줄어들게 된다: 700 - dim = xmove
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), dsize=(dim,dim))
        img_id = ((img_path.split('/')[-1]).split('_')[-1]).split('.')[0]
        height, width = img.shape[:2]
        
        blank_image = np.zeros((700,700,3), np.uint8)
        blank_image[:,:] = (255,255,255)
        l_img = blank_image.copy()
        for xmove in x_offset:
            if dim + xmove < 700:
                for row in range(len(img)):
                    for col in range(len(img)):
                        l_img[row, col+xmove] = img[row, col]
                        
                acupuncture_id = f"{acupuncture_info}_{img_id}"
                acupuncture_new_id = acupuncture_id + f'_dim{dim}_scaling{xmove}'
                x, y, xy = json_data[acupuncture_id][1].values()
                new_x, new_y = x * dim / 700 + xmove, y * dim / 700
                new_xy = (new_x, new_y)

                json_data[acupuncture_new_id] = list()
                json_data[acupuncture_new_id].append({
                    "acup_info": f"{acupuncture_info}",
                    "hand_pos": f"{img_hand_pos}",
                    "acup_size": f"{acupuncture_size}"
                })
                json_data[acupuncture_new_id].append({
                    "acup_coord_x": new_x,
                    "acup_coord_y": new_y,
                    "acup_coord": new_xy
                })

                cv2.imwrite(f'./{save_directory}/{acupuncture_new_id}.png', l_img)
                l_img = blank_image.copy()
    save_json_file(json_data)

##############################################################################################

# 혈점 정보, 점 사이즈, 손 위치 입력
acupuncture_info = input('혈자리를 입력해주세요. ex) 소충 ')

##############################################################################################

# TODO: transformation(ongoing)

x_moves = [-60, -30, 30, 60]
dimensions = [400,500]
angles = [-45, -30, 15, 15, 30, 45]

acupuncture_size = 3
acupuncture_db = open_temp_db()
acupuncture_info = is_acupuncture(acupuncture_info, acupuncture_db)

hand_path_frst = f'./{acupuncture_info}/{acupuncture_info}_dorsal_left/'
hand_path_scnd = f'./{acupuncture_info}/{acupuncture_info}_dorsal_right/'
hand_path_thrd = f'./{acupuncture_info}/{acupuncture_info}_palmar_left/'
hand_path_frth = f'./{acupuncture_info}/{acupuncture_info}_palmar_right/'

temp_paths = [hand_path_frst, hand_path_scnd, hand_path_thrd, hand_path_frth]
changed_hands_path = [p+'change' for p in temp_paths if isdir(p)]

json_file = "./json_data/{}_info.json".format(acupuncture_info)
json_data = open_json_file(json_file)

images_info = sorted(make_path_tuple(acupuncture_info, changed_hands_path), key=lambda x:x[1])

for i in range(len(x_moves)):
    translate_image(images_info, json_data, x_moves[i])
for i in range(len(dimensions)):
    scaling_image(images_info, json_data, dimensions[i])
# for i in range(len(angles)):
#     rotate_image(images_info, json_data, angles[i])