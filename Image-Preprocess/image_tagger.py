from os import listdir, makedirs
from os.path import isfile, isdir, join
import cv2  
import copy
import json

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

def get_hand_position(hand_pos):
    hand_position = ['dorsal_right', 'dorsal_left', 'palmar_right', 'palmar_left']
    hand_position = hand_position[hand_pos-1]
    return hand_position

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
        return dict()        

def save_json_file(json_data):
    with open(json_file, 'w') as outfile:
        json.dump(json_data, outfile, indent='\t')

def draw_circle(event, x, y, flags, param):     
    global img
    global img_org
    global img_path
    global acupuncture_id
        
    k = cv2.waitKey(10) & 0xFF    

    # 'c' => 마우스 커서 중심으로 1:1 비율 Cropping
    if k == ord('c'):
        # print("cropping the image, center at : {}".format((x,y)))
        x_center = x
        img_max_pixl = img.shape[1]
        img_min_pixl = 0
        
        crop_half = img.shape[0]/2
        x_l_bndry = x_center - crop_half
        x_r_bndry = x_center + crop_half

        if x_l_bndry < img_min_pixl:
            x_center += img_min_pixl - x_l_bndry

        elif x_r_bndry > img_max_pixl:
            x_center -= x_r_bndry - img_max_pixl

        x_l_bndry = int(x_center - crop_half)
        x_r_bndry = int(x_center + crop_half) 
        img = img[:, x_l_bndry:x_r_bndry] 
        img_org = copy.deepcopy(img)


    # 왼쪽 버튼 클릭 => 점 찍기
    elif event == cv2.EVENT_LBUTTONDOWN: 
        print("coordinate : {}".format((x,y)))
        cv2.circle(img, (x, y), acupuncture_size, (255, 0, 0), -1) 
        if len(json_data[acupuncture_id]) < 2:
            json_data[acupuncture_id].append({
                "acup_coord_x": x,
                "acup_coord_y": y,
                "acup_coord": (x,y)
            })
        else:
            json_data[acupuncture_id][1] = {
                "acup_coord_x": x,
                "acup_coord_y": y,
                "acup_coord": (x,y)
            }
        # print(json_data[acupuncture_id][1])

    # 오른쪽 버튼 클릭 => 점 지우기
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), dsize=(700,700))
        img = cv2.rotate(img, cv2.ROTATE_180)
        print("cleaned")
          
def show_images(image_files, json_data):

    global img
    global img_org
    global img_path
    global acupuncture_id

    for i in range(0, len(image_files)):
        img_path = join(mypath,image_files[i])
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), dsize=(700,700))
        img = cv2.rotate(img, cv2.ROTATE_180)
        img_org = copy.deepcopy(img)
        print(img.shape[0], img.shape[1], img.shape[2])

        img_name = img_path.split('/')[-1]
        img_id_dtype = img_name.split('_')[-1]
        img_id = img_id_dtype.split('.')[0]

        acupuncture_id = f"{acupuncture_info}_{img_id}"
        
        json_data[acupuncture_id] = list()
        json_data[acupuncture_id].append({
            "acup_info": f"{acupuncture_info}",
            "hand_pos": f"{hand_position}",
            "acup_size": f"{acupuncture_size}"
        })
        # print(json_data[acupuncture_id][0])
        while(1):
            cv2.setMouseCallback(img_path, draw_circle)
            cv2.imshow(img_path, img)
            cv2.moveWindow(img_path, 0,0)
            k = cv2.waitKey(10) & 0xFF
            
            # ESC => 저장 안하고 다음사진으로 넘어가기
            if k == 27:
                json_data.pop(acupuncture_id)
                cv2.destroyAllWindows()
                break

            # 's' => 원본 사진과 점찍은 사진 저장하고 다음사진으로 넘어가기
            elif k == ord('s') and len(json_data[acupuncture_id]) == 2:
                # 점찍은 사진 저장하기 위한 directory/filename 설정
                # 예시: cv2.imwrite('./directory/name_{}.jpg'.format(img_id), img)
                cv2.imwrite('./{0}/change/{1}_{2}.png'.format(save_directory, acupuncture_info, img_id),img) 
                
                # 원본 사진 저장하기 위한 dirctory/filename 설정
                cv2.imwrite('./{0}/org/Hand_{1}.png'.format(save_directory, img_id), img_org) 
                
                break

            # 'z' => 프로그램 종료(terminate program)
            elif k == ord('z'):
                json_data.pop(acupuncture_id)
                save_json_file(json_data)
                exit()
                
        cv2.destroyAllWindows()

    save_json_file(json_data)

##############################################################################################

# 이미지를 불러올 directory 폴더 설정, 적당한 이미지 수만큼 넣고 돌릴것 or (cv::OutOfMemoryError)
mypath = './test2'

# 혈점 정보, 점 사이즈, 손 위치 입력
acupuncture_info = input('혈자리를 입력해주세요. ex) 소충 ')
acupuncture_size = 3

# 손모양 => 'dorsal_right':1, 'dorsal_left':2, 'palmar_right':3, 'palmar_left':4
hand_p = {'dorsal_right':1, 'dorsal_left':2, 'palmar_right':3, 'palmar_left':4}
hand_pos = hand_p[input('손의 위치를 입력해주세요. ex) palmar_left ')]

##############################################################################################

hand_position = get_hand_position(hand_pos)

acupuncture_db = open_temp_db()
acupuncture_info = is_acupuncture(acupuncture_info, acupuncture_db)

save_directory = f'{acupuncture_info}/{acupuncture_info}_{hand_position}'

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

json_file = "./json_data/{}_info.json".format(acupuncture_info)
json_data = open_json_file(json_file)
image_files = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
show_images(image_files, json_data)