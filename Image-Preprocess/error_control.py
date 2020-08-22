from os import listdir, makedirs
from os.path import isfile, isdir, join
import json
import cv2
import numpy as np
import re

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
def inter_check(save_directory, change_dir, org_dir, acup):
    
    str_ = re.compile(r'[a-zA-Z]')
    m = re.compile(r'[_,.]')
    
    # using set, searching intersection file
    image_files_change = list( f for f in listdir(change_dir) if isfile(join(change_dir,f)))
    image_files_org = list( f for f in listdir(org_dir) if isfile(join(org_dir,f)))
    
    change = set(m.sub("", str_.sub('',image_files_change[i])) for i in range(len(image_files_change)))
    org = set(m.sub("", str_.sub('',image_files_org[i])) for i in range(len(image_files_org)))
    
    stan = change & org
    # make dir
    try:
        if not(isdir(save_directory)):
            makedirs(join(save_directory))
        if not(isdir(save_directory+'/change_')):
            makedirs(join(save_directory+'/change_'))
        if not (isdir(save_directory+'/org_')):
            makedirs(join(save_directory+'/org_'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # getting intersection image
    #if len(image_files_change) != len(image_files_org):
    tmp = list(stan)

    for i in range(len(tmp)):
        tmp_img = cv2.imread(save_directory + '/change/' + acup + '_' + tmp[i]+'.png')
        cv2.imwrite(save_directory+ '/change_/' + acup + '_' + tmp[i] + '.png', tmp_img)

    for i in range(len(tmp)):
        tmp_img = cv2.imread(save_directory + '/org/' + 'Hand_' + tmp[i] + '.png')
        cv2.imwrite(save_directory+ '/org_/'  + 'Hand_' + tmp[i] + '.png', tmp_img)

def image_quality_check(save_directory, change_dir, org_dir):
    
    image_files_change = list( f for f in listdir(change_dir) if isfile(join(change_dir,f)))
    image_files_org = list( f for f in listdir(org_dir) if isfile(join(org_dir,f)))

    image_files_change.sort()
    image_files_org.sort()
    
    image_files_change_set = set(image_files_change)
    image_files_org_set = set(image_files_org)
    
    # if the image after previous image is not the same, delete the image.

    for i in range(len(image_files_change)):
        
        tmp = np.array(cv2.imread(change_dir + '/' + image_files_change[i])) - np.array(cv2.imread(org_dir + '/' + image_files_org[i]))
        x, y, z= np.where(tmp > 0)
        # if the point is not 6, image is not acceptable.

        if len(set(x)) != 7 or len(set(y)) != 7:
            image_files_change_set -= {image_files_change[i]}


    for i in range(len(image_files_org)):
        
        tmp = np.array(cv2.imread(change_dir + '/' + image_files_change[i])) - np.array(cv2.imread(org_dir + '/' + image_files_org[i]))
        x, y, z= np.where(tmp > 0)

        # if the point is not 6, image is not acceptable.
        if len(set(x)) != 7 or len(set(y)) != 7:
            image_files_org_set -= {image_files_org[i]}
    

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
        tmp_img_ch = cv2.imread(change_dir + '/' + res_ch[i])
        cv2.imwrite(save_directory + '/change_qc/' + res_ch[i], tmp_img_ch)
        
        tmp_img_org = cv2.imread(org_dir + '/' + res_org[i])
        cv2.imwrite(save_directory + '/org_qc/' + res_org[i], tmp_img_org)

def make_json(save_dir, change_left, org_left, change_right, org_right, info_left, info_right):
    
    image_files_change_left = list( f for f in listdir(change_left) if isfile(join(change_left,f)))
    image_files_org_left = list( f for f in listdir(org_left) if isfile(join(org_left,f)))
    
    image_files_change_right = list( f for f in listdir(change_right) if isfile(join(change_right,f)))
    image_files_org_right = list( f for f in listdir(org_right) if isfile(join(org_right,f)))
    acup_name = info_left['acup_info']
    acup = {}
    image_files_change_left.sort()
    image_files_org_left.sort()
    image_files_change_right.sort()
    image_files_org_right.sort()
    
    for i in range(len(image_files_change_left)):
        diff = np.array(cv2.imread(change_left + image_files_change_left[i])) - np.array(cv2.imread(org_left + image_files_org_left[i]))
        x,y,z=np.where(diff > 0)
        y_, x_= 3 + x.min() , 3 + y.min()
        tmp = {}
        acup[image_files_change_left[i][:-4]] = [info_left, {"acup_coord_x": x_, "acup_coord_y": y_, "acup_coord": [x_, y_]}]
        #print(image_files_change[i])
        
    for i in range(len(image_files_change_right)):
        diff = np.array(cv2.imread(change_right + image_files_change_right[i])) - np.array(cv2.imread(org_right + image_files_org_right[i]))
        x,y,z=np.where(diff > 0)
        y_, x_= 3 + x.min() , 3 + y.min()
        tmp = {}
        acup[image_files_change_right[i][:-4]] = [info_right, {"acup_coord_x": x_, "acup_coord_y": y_, "acup_coord": [x_, y_]}]
    print(acup)
    with open(image_dir + acup_name + '_info_수정.json', 'w') as fp:
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
image_dir = input('이미지 디렉터리를 입력하시오')
pos = input('palmar or dorsal?')
info_left['acup_info'] = acup
info_right['acup_info'] = acup

####################################################################################################
# checking intersection & quaility
save_dir = image_dir+acup+'/'+acup+'_'+pos+'_left'
print(save_dir)
change_dir = image_dir+acup+'/'+acup+'_'+pos+'_left/change'
org_dir = image_dir+acup+'/'+acup+'_'+pos+'_left/org'
inter_check(save_dir, change_dir, org_dir, acup)
print('left inter check finish')
image_quality_check(save_dir, change_dir +'_', org_dir + '_')
print('left quality check finish')

save_dir = image_dir+acup+'/'+acup+'_'+pos+'_right'
change_dir = image_dir+acup+'/'+acup+'_'+pos+'_right/change'
org_dir = image_dir+acup+'/'+acup+'_'+pos+'_right/org'
inter_check(save_dir, change_dir, org_dir, acup)
print('right inter check finish')
image_quality_check(save_dir, change_dir +'_', org_dir + '_')
print('right quality check finish')

print('확인 해주세요.')

# make new json
change_left = image_dir+acup+'/'+acup+'_'+pos+'_left/change_qc/'
org_left = image_dir+acup+'/'+acup+'_'+pos+'_left/org_qc/'
change_right = image_dir+acup+'/'+acup+'_'+pos+'_right/change_qc/'
org_right = image_dir+acup+'/'+acup+'_'+pos+'_right/org_qc/'
make_json(image_dir, change_left, org_left, change_right, org_right, info_left, info_right)
