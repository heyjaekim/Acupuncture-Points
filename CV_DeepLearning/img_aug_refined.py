from os import listdir, makedirs
from os.path import isfile, isdir, join
import cv2, copy, json, os
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os, shutil, PIL 

##################################################################
# directory

dir1 = './background-images/'
imdirs = os.listdir(dir1)
imlist = [ plt.imread(dir1 + i) for i in imdirs ]


##################################################################
# open json file


def open_json_file(file_path):
    if isfile(file_path):
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)
        return json_data
    else:
        print("json file doesn't exist")
        exit()        

#####################################################################################
# clearning folders 
def move_dirs(oslist, src, dst):
    for i in oslist:
        shutil.move(os.path.join(src, i), dst)


def clean_and_rename_directory( dirlist ):
    # directory must be './CV_DeepLearning/Acu_Dataset'
    if os.getcwd().split('/')[-1] == 'Image-Preprocess':
        print('cur')
        os.chdir('../CV_DeepLearning')
    
    dir_lists = os.listdir(dirlist)
    print(os.getcwd())
    for dir_name in dir_lists:
        dir_cur = './Acu_Dataset/' + dir_name
        # 폴더가 하나인 경우 상위 폴더로 통합 
        if len(os.listdir(dir_cur) ) == 1:
            dir_cur2 = dir_cur + '/' + dir_name
            folders = os.listdir(dir_cur2)
            for j in folders:
                shutil.move ( dir_cur2 + '/' + str(j) , dir_cur)
            shutil.rmtree(dir_cur2)
            print('cleaning directory ', dir_cur2)

        if len(os.listdir(dir_cur) ) == 3:
            os.mkdir(dir_cur + '/org')
            if True in [ 'dorsal' for d in os.listdir(dir_cur)]: 
                kw = '_dorsal_'
            else: 
                kw = '_palmar_'

            left_dir = dir_cur + '/'+ dir_name + kw + 'left'
            right_dir = dir_cur + '/'+ dir_name + kw + 'right'
            left_list = os.listdir(left_dir)
            right_list = os.listdir(right_dir)
            move_dirs(left_list, left_dir, dir_cur + '/org')
            move_dirs(right_list, right_dir, dir_cur + '/org')

            shutil.rmtree(left_dir)
            shutil.rmtree(right_dir)
            print('completed cleaning folders: ', dir_cur)

            np_lst = np.array(os.listdir(dir_cur))
            old_name = np_lst[np.array([(not 'org' in d) for d in os.listdir(dir_cur)])][0]

            old_file = os.path.join( dir_cur, old_name )
            new_file = os.path.join( dir_cur, dir_name + '_info.json'  )
            print('oldfile: ', old_file, ' new_file : ', new_file)
            os.rename(old_file, new_file)



            
#####################################################################################
# DB & Path Check

def open_temp_db():
    db = []
    dorsal_acupunctures = {'양계':'yanggye', '양지':'yangji', '외관':'oegwan', '양곡':'yanggok', 
                            '합곡':'hapgok', '중저':'jungjer', '삼간':'samgan', '이간':'egan',
                            '액문':'ekmoon','상양':'sangyang','중층':'jungcheung','소충':'sochung',
                            '소택':'sotack','관충':'gwanchung', '휴게': 'hugye'}
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
        
#####################################################################################
# Image processing: crop, clear background, fill background, rotate, translate, scale
def create_circle_patch(img, x, y, color, r=1):
    '''
    input : 0~1 float32 ndarray
    output : 0~255 uint8 ndarray with dots
    '''
    # img = (img * 255 / np.max(img)).astype('uint8')
    if type(img) == np.ndarray:
        img = Image.fromarray(img.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(img)
    x = 2 if x < 0 else x
    y = 2 if y < 0 else y

    leftUpPoint = (x-r, y-r)
    rightDownPoint = (x+r, y+r)

    twoPointList = [leftUpPoint, rightDownPoint]
    if color == 'blue':
        fill = (0,0,255,0)
    if color == 'red':
        fill = (255,0,0,0)
    draw.ellipse(twoPointList, fill=fill)
    img = np.array(img)
    img = (img  / np.max(img))
    return img


def get_img_distn (img, kw='orange'):
    '''
    W x H x C ; C: RGB
    '''
    if kw == 'orange':
        _ = plt.hist(img.ravel(), bins = 256, color = 'orange', )
    
    if kw == 'red':
        _ = plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

    if kw == 'green':
        _ = plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    
    if kw == 'blue':
        _ = plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

    plt.show()
    


def rand_crop_img (img, size = 250, change_col = True, *args):
    '''
    gets nd array greater than (256x256) as an input
    returns randomly cropped image (256x256) as an output
    channel values are uint of 0 to 255
    '''
    if img.dtype == 'float32':
        img = (img * 255 / np.max(img)).astype('uint8')
    im_pil = Image.fromarray(img.astype('uint8'), 'RGB')

    x_c = np.random.randint(0, im_pil.size[0] - size)
    y_c = np.random.randint(0, im_pil.size[1] - size)
    img_cropped = im_pil.crop((x_c, y_c, x_c + size, y_c + size)) 
    img_cropped = np.asarray(img_cropped) / 255.0

    if change_col == True:
        # change color distn 
        for j in range(3):
            img_cropped[:, :, j] = (img_cropped[:, :, j] + np.random.uniform(0, 0.15) ) / 2.0
            img_cropped = img_cropped / np.max(img_cropped)

    return img_cropped


def clear_background(img):
    # img = cv2.imread('test2\Hand_0000002.png')

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_range = (0,0,100)
    upper_range = (358,45,255)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # invert mask
    mask = 255 - mask
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # apply morphology closing and opening to mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = img.copy()
    result[mask==0] = (255,255,255)
    
    return result
    

def img_from_mask2(img, mask ):
    '''
    mask : 2d array of true/false
    img : 3d array with rgb channels
    '''
    r = img[:,:,0] + 255*mask[:,:,0] 
    g = img[:,:,1] + 255*mask[:,:,1]  
    b = img[:,:,2] + 255*mask[:,:,2]  
    rgb = np.dstack((r,g,b))
    rgb[rgb > 255] = 0
    return rgb


def save_json_file2(json_data, json_file):
    with open(json_file, 'w') as outfile:
        json.dump(json_data, outfile, indent='\t')

def rotate_bound(image, theta):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_TRANSPARENT)

def rotate_box(coord, cx, cy, h, w, theta):
    
    # opencv calculates standard transformation matrix
    M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
    
    # Grab  the rotation components of the matrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    
    # Prepare the vector to be transformed
    v = [coord[0], coord[1],1]
    
    # Perform the actual rotation and return the image
    calculated = np.dot(M,v)
    new_bb = (calculated[0], calculated[1])
    
    return new_bb
        
def fill_background_image2(org_path, json_data, save_name, save_path, imlist, chg_flag = False):
    '''
    img_info : list of image names (os.listdir) in the original path 
    '''
    img_info = os.listdir(org_path)
    # acup info
    acu_info = list(json_data)[0].split('_')[0]
    # directory for saving image
    save_directory = save_path + '/' + save_name
    if not(isdir(save_directory)):
        makedirs(save_directory)
    img_directory = save_directory + '/' + save_name 
    if not(isdir(img_directory)):
        makedirs(img_directory)
    # json_file
    json_new = dict()
    # random index
    random_indx = np.random.randint(low = 0,high = len(imlist), size = len(img_info))
    
    for i in range(0, len(img_info)):
        img_path = org_path + '/' + img_info[i]
        #img_id = img_info[i].split('_')[-1].split('.')[0]
        #img_id = name_list[1]
        names = img_info[i].split('_')
        img_id = names[1].split('.')[0]
        
        if chg_flag == True:
            chg_id = names[2].split('.')[0]
            acupuncture_id = f"{acu_info}_{img_id}_{chg_id}"
        else: 
             acupuncture_id = f"{acu_info}_{img_id}"
        
        # json tag
        acupuncture_new_id = acupuncture_id + '_'+save_name
        xy = json_data[acupuncture_id][1]['acup_coord']
        img_hand_pos =  json_data[acupuncture_id][0]['hand_pos']
        
        json_new[acupuncture_new_id] = list()
        json_new[acupuncture_new_id].append({
            "acup_info": f"{acu_info}",
            "hand_pos": f"{img_hand_pos}"
        })
        json_new[acupuncture_new_id].append({
            "acup_coord": xy
        })
        
        # read the image file, then generate image
        img = clear_background(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        height, width = img.shape[:2]
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # np format

        # get img mask
        im_mask1 = 1* (im_rgb != 255)
        im_mask2 = 1* (im_rgb == 255)
        
        npimg = rand_crop_img( imlist[ random_indx[i]])
        uint_img = (npimg * 255 / np.max(npimg)).astype('uint8')
        b_image = cv2.resize( uint_img , dsize=(height,width)) 
        
        new_img = img_from_mask2( b_image, im_mask1 ) +  img_from_mask2( im_rgb, im_mask2 )
        # get PIL img
        img_pil = Image.fromarray(new_img.astype('uint8'), 'RGB')   
        img_pil.save(img_directory + '/' + acupuncture_new_id + '.png')
        
        if i % 10 == 0:
            print('completed ' + str(i) + ' images to fill')
    save_json_file2(json_new, save_directory + '/' + acu_info + '_' + save_name + '.json')

    
def rotate_image_trnsfm(org_path, json_data, save_name, save_path, chg_flag = False):
    
    img_info = os.listdir(org_path)
    acu_info = list(json_data)[0].split('_')[0]
    
    save_directory = save_path + '/' + save_name
    if not(isdir(save_directory)):
        makedirs(save_directory)

    img_directory = save_directory + '/' + save_name 
    if not(isdir(img_directory)):
        makedirs(img_directory)
    # json_file
    json_new = dict()
    
    angle_list = np.random.randint(low = -180, high = 180, size = len(img_info))
        
    for i in range(0, len(img_info)):
        img_path = org_path + '/' + img_info[i]
        names = img_info[i].split('_')
        img_id = names[1].split('.')[0]
        angle = angle_list[i]

        if chg_flag == True:
            chg_id = names[2].split('.')[0]
            acupuncture_id = f"{acu_info}_{img_id}_{chg_id}"
        else: 
            acupuncture_id = f"{acu_info}_{img_id}"

        # json tag
        acupuncture_new_id = acupuncture_id + f'_rt{angle}'

        x, y, xy = json_data[acupuncture_id][1].values()
        img_hand_pos =  json_data[acupuncture_id][0]['hand_pos']
        
        # read the image file, then generate rotated image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        rotated_img = rotate_bound(img, angle)

        # height and width for images
        height, width = img.shape[:2]
        (cx, cy) = (width // 2, height // 2)
        (new_height, new_width) = rotated_img.shape[:2]
        (new_cx, new_cy) = (new_width // 2, new_height // 2)
        
        new_x, new_y = rotate_box((x, y), cx, cy, height, width, angle)
        
        # change to 700 x 700 
        new_x, new_y = int(new_x * 700 / new_width ), int(new_y * 700 / new_height )
        new_xy = (new_x, new_y)
        

        json_new[acupuncture_new_id] = list()
        json_new[acupuncture_new_id].append({
            "acup_info": f"{acu_info}",
            "hand_pos": f"{img_hand_pos}"
        })
        json_new[acupuncture_new_id].append({
            "acup_coord": new_xy
        })
        
        im_mask1 = 255 * (rotated_img == 0)
        rotated_img = (im_mask1 + rotated_img).astype('uint8')
        # resize to 700 x 700
        rotated_img = cv2.resize( rotated_img , dsize=(700,700)) 
        
        cv2.imwrite(f'./{img_directory}/{acupuncture_new_id}.png', rotated_img)
        if i % 100 == 0:
            print('completed ' + str(i) + ' images to rotate')

    save_json_file2(json_new, save_directory + '/' + acu_info + '_' + save_name + '.json')


# translation & cropping related
def margin_finder(img):
    # input must be rgb immage
    #width, height = img.shape[:2]
    left_margin = int(np.mean( [ (img[:,:,i] -255).max(axis = 0).argmax() for i in range(3)])) 
    up_margin = int(np.mean( [ (img[:,:,i] -255).max(axis = 1).argmax() for i in range(3)])) 
    
    r = (img[:,:,0]).min(axis = 0)
    g = (img[:,:,1]).min(axis = 0)
    b = (img[:,:,2]).min(axis = 0)
    right_margin =5 + min(np.where( ((r >= 200) & (r<235)  & (g>=238) & (g<250) & (b >= 225) & (b < 250)))[0])
    return left_margin, right_margin, up_margin

def img_crop(img, margin):
    left_margin, right_margin, up_margin = margin
    return (img[ up_margin:, left_margin:right_margin, :])


################
# Accessing Files
def scale_and_translate_img (org_path, json_data, save_name, save_path, chg_flag = False):
    img_info = os.listdir(org_path)
    # acup info
    acu_info = list(json_data)[0].split('_')[0]
    # directory for saving image
    save_directory = save_path + '/' + save_name
    if not(isdir(save_directory)):
        makedirs(save_directory)
    
    img_directory = save_directory + '/' + save_name 
    if not(isdir(img_directory)):
        makedirs(img_directory)
    # json_file
    json_new = dict()
    
    for i in range(0, len(img_info)):
        img_path = org_path + '/' + img_info[i]
        names = img_info[i].split('_')
        img_id = names[1].split('.')[0]

        if chg_flag == True:
            chg_id = names[2].split('.')[0]
            acupuncture_id = f"{acu_info}_{img_id}_{chg_id}"
        else: 
            acupuncture_id = f"{acu_info}_{img_id}"

        # json tag

        x_org, y_org, xy = json_data[acupuncture_id][1].values()
        img_hand_pos =  json_data[acupuncture_id][0]['hand_pos']

        pil_im = PIL.Image.open(img_path)
        np_im = np.array(pil_im)

        ###############################################
        # processing
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blurred, 50, 100, 255)

        # edge points location 
        y_vals = np.where(edged.max(axis=1))[0]
        x_vals = np.where(edged.max(axis=0))[0]
        # cropping boundary
        y_high = np.min(y_vals)
        y_low = np.max(y_vals)
        x_high = np.max(x_vals)
        x_low = np.min(x_vals)
        # coordinate after cropping: 
        x_new = x_org - x_low
        y_new = y_org - y_high
        # cropped image
        cropped = np_im[y_high:y_low, x_low:x_high, :]
        cp_height, cp_width = cropped.shape[:2]

        # Scaling IMG
        scale_prop = 0.7
        scalar = np.random.uniform(scale_prop,1)
        x_scale = int(scalar * np.random.randint(cp_width/1.15, min(cp_width*1.5, 700)))
        y_scale = int(scalar * np.random.randint(cp_height/1.5, 700))
        # x,y coord after scaling
        x_coord_sc = x_new * x_scale / cp_width
        y_coord_sc = y_new * y_scale / cp_height
        # scaled 
        pil = Image.fromarray(cropped.astype('uint8'), 'RGB')
        pil = pil.resize(size = (x_scale, y_scale))
        np_scaled = np.array(pil)
        
        # Translating IMG
        move_x = np.random.randint(x_scale, 700)
        x_left = move_x - x_scale
        x_right = 700 - move_x 

        move_y = np.random.randint(y_scale, 700)
        y_up = move_y - y_scale
        y_down = 700 - move_y

        # add to blank 700x700
        left_margin = (255*np.ones(shape = (y_scale,x_left, 3), dtype = 'uint8'))
        right_margin = (255*np.ones(shape = (y_scale,x_right, 3), dtype = 'uint8'))
        up_margin =  (255*np.ones(shape = (y_up, 700, 3), dtype = 'uint8'))
        down_margin =  (255*np.ones(shape = ( y_down,700, 3), dtype = 'uint8'))
        stack1 = np.hstack((left_margin, np_scaled, right_margin))
        fin_img = np.vstack((up_margin, stack1, down_margin))

        # new coordinate after translation 
        x_new_tr = int(x_left + x_coord_sc)
        y_new_tr = int(y_up + y_coord_sc)
        new_xy = x_new_tr, y_new_tr

        acupuncture_new_id = acupuncture_id + f'_sctr{x_scale}+{y_scale}+{x_left}+{y_up}'

        json_new[acupuncture_new_id] = list()
        json_new[acupuncture_new_id].append({
        "acup_info": f"{acu_info}",
        "hand_pos": f"{img_hand_pos}"
        })
        json_new[acupuncture_new_id].append({
        "acup_coord": new_xy
        })

        # Save new img
        img_pil = Image.fromarray(fin_img.astype('uint8'), 'RGB')   
        img_pil.save(img_directory + '/' + acupuncture_new_id + '.png')

        if i % 100 == 0:
            print('completed ' + str(i) + ' images to sctr')
    save_json_file2(json_new, save_directory + '/' + acu_info + '_' + save_name + '.json')

##########################################################################################

# Augment


def augment_hands( class_label, save_name= 'rotated', img_folder = 'org', json_name = None, imlist = None, chg_flag = False):
    #  'C:\\Users\\NormalKim\\[0_BigData_Acu]\\Acupuncture-Points\\CV_DeepLearning'
    if chg_flag == False: 
        org_path = f'./Acu_Dataset/{class_label}/{img_folder}' # rotated/rotated
        json_data = open_json_file(f'./Acu_Dataset/{class_label}/{class_label}_info.json')
        save_path = f'./Acu_Dataset/{class_label}/'
    else:
        folder_path = f'./Acu_Dataset/{class_label}/{img_folder}' # img_folder = rotated
        org_path = folder_path + f'/{img_folder}'
        json_data = open_json_file(folder_path + f'/{json_name}.json')
        save_path = f'./Acu_Dataset/{class_label}/'
        
    print('starting: ', class_label , '  ', save_name)
    if  'rotated' in save_name.split('_')[-1]:
        rotate_image_trnsfm(org_path, json_data, save_name, save_path, chg_flag)
    
    elif 'filled' in save_name.split('_')[-1]:
        fill_background_image2( org_path, json_data, save_name, save_path, imlist, chg_flag)

    elif 'sctr' in save_name.split('_')[-1]:
        scale_and_translate_img( org_path, json_data, save_name, save_path, chg_flag)













