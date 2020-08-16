from os import listdir, makedirs
from os.path import isfile, isdir, join
import json
import cv2
import numpy as np

# for writing json in window10
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# intersection check and get acceptable file
def inter_check(save_directory, change_dir, org_dir):
    
    # using set, searching intersection file
    image_files_change = set( f for f in listdir(change_dir) if isfile(join(change_dir,f))
    image_files_org = set( f for f in listdir(org_dir) if isfile(join(org_dir,f)))
    stan = image_files_change & image_files_change

    # make dir
    try:
        if not(isdir(save_directory)):
            makedirs(join(save_directory))
        if not(isdir(save_directory+'/change')):
            makedirs(join(save_directory+'/change'))
        if not (isdir(save_directory+'/org')):
            makedirs(join(save_directory+'/org'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # getting intersection image
    if len(image_files_change) != len(stan):
        tmp = list(stan)
        for i in range(len(tmp)):
            tmp_img = cv2.imread(tmp[i])
            cv2.imwrite(save_directory+ '/change' + tmp[i], tmp_img)

    if len(image_files_org) != len(stan):
        tmp = list(stan)
        for i in range(len(tmp)):
            tmp_img = cv2.imread(tmp[i])
            cv2.imwrite(save_directory+ '/org' + tmp[i], tmp_img)

def image_quality_check(save_dir, change_dir, org_dir):
    image_files_change = list( f for f in listdir(change_dir) if isfile(join(change_dir,f))
    image_files_org = list( f for f in listdir(org_dir) if isfile(join(org_dir,f)))

    image_files_change.sort()
    image_files_org.sort()

    image_files_change_set = set(image_files_change)
    image_files_org_set = set(image_files_org)

    # if the image after previous image is not the same, delete the image.
    for i in range(len(image_files_change)):
        
        tmp = np.array(cv2.imread(image_files_change[i])) - np.array(cv2.imread(image_files_org[i]))
        x, y, z= np.where(tmp > 0)

        # if the point is not 6, image is not acceptable.
        if len(set(x)) != 6 or len(set(y)) != 6:
            image_files_change_set -= {image_files_change[i]}

    for i in range(len(image_files_org)):
        
        tmp = np.array(cv2.imread(image_files_change[i])) - np.array(cv2.imread(image_files_org[i]))
        x, y, z= np.where(tmp > 0)

        # if the point is not 6, image is not acceptable.
        if len(set(x)) != 6 or len(set(y)) != 6:
            image_files_org_set -= {image_files_org[i]}
    
    if len(res_ch) == len(res_org):

        res_ch = list(image_files_change_set)
        res_org = list(image_files_org_set)

    # make dir / named QC (Quaility Control)
    try:
        if not(isdir(save_directory)):
            makedirs(join(save_directory))
        if not(isdir(save_directory+'/change_qc')):
            makedirs(join(save_directory+'/change_qc'))
        if not (isdir(save_directory+'/org_qc')):
            makedirs(join(save_directory+'/org_qc'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    for i in range(len(res_ch)):
        tmp_img = cv2.imread(tmp[i])
        cv2.imwrite(save_directory+ '/change_qc' + tmp[i], tmp_img)
        cv2.imwrite(save_directory+ '/org_qc' + tmp[i], tmp_img)

def make_json(save_dir, change_left, org_left, change_right, org_right):
    
    image_files_change_left = list( f for f in listdir(change_left) if isfile(join(change_left,f))
    image_files_org_left = list( f for f in listdir(org_left) if isfile(join(org_left,f)))
    
    image_files_change_right = list( f for f in listdir(change_right) if isfile(join(change_right,f))
    image_files_org_right = list( f for f in listdir(org_right) if isfile(join(org_right,f)))

    acup = {}
    image_files_change.sort()
    image_files_org.sort()

    for i in range(len(image_files_change)):
        diff = np.array(cv2.imread(change + image_files_change_left[i])) - np.array(cv2.imread(org + image_files_org_left[i]))
        x,y,z=np.where(diff > 0)
        y_, x_= 3 + x.min() , 3 + y.min()
        tmp = {}
        acup[image_files_change[i][:-4]] = [info_left, {"acup_coord_x": x_, "acup_coord_y": y_, "acup_coord": [x_, y_]}]
    
    for i in range(len(image_files_change)):
        diff = np.array(cv2.imread(change + image_files_change_right[i])) - np.array(cv2.imread(org + image_files_org_right[i]))
        x,y,z=np.where(diff > 0)
        y_, x_= 3 + x.min() , 3 + y.min()
        tmp = {}
        acup[image_files_change[i][:-4]] = [info_left, {"acup_coord_x": x_, "acup_coord_y": y_, "acup_coord": [x_, y_]}]

    with open(acup + '_info_수정.json', 'w') as fp:
        json.dump(acup, fp, cls = NpEncoder)



####################################################################################################
# information
info_left = {
            "acup_info": "sotack",
            "hand_pos": "dorsal_left",
            "acup_size": "3"
        }
info_right = {
            "acup_info": "sotack",
            "hand_pos": "dorsal_right",
            "acup_size": "3"
        }

acup = input("부위를 입력하시오.")

info_left['acup_info'] = acup
info_right['acup_info'] = acup

####################################################################################################
# checking intersection
save_dir = input("저장하려고 하는 디렉터리를 입력하시오.")
change_dir = input('바뀐 디렉터리를 입력하시오.')
org_dir = input('바뀌기 전 디렉터리를 입력하시오')
inter_check(save_dir, change_dir, org_dir)
# quality check
image_quality_check(save_dir, change_dir, org_dir)
# json
make_json(save_dir, change_dir, org_dir)
