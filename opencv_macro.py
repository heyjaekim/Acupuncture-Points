from os import listdir
from os.path import isfile, join
import numpy as np
import cv2  

def draw_circle(event, x, y, flags, param):       
    if event == cv2.EVENT_LBUTTONDOWN: 
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1) 
          
def show_image(img, img_org, img_path):
    cv2.namedWindow(winname = img_path) 
    cv2.setMouseCallback(img_path, draw_circle)

    while(1):
        cv2.imshow(img_path, img) 
        cv2.moveWindow(img_path, 40,30)
        k = cv2.waitKey(10) & 0xFF
        
        # ASCII value (27) => ESC
        if k == 27:
            cv2.destroyAllWindows()
            break

        # ASCII value (99) => 'c'
        # 'c' 누르면 저장되고 다음 사진으로 넘어감
        elif k == ord('c'):
            img_path = img_path.split('/')[-1]
            img_id_with_dtype = img_path.split('_')[-1]
            img_id = img_id_with_dtype.split('.')[0]
            # 원하는 directory/filename 설정
            # cv2.imwrite('./directory/name_{}.jpg'.format(img_id), img)
            cv2.imwrite('./sangyang_dorsal_left/sangyang_{}.png'.format(img_id),img) 
            # 원본 사진 png 형태로 저장
            cv2.imwrite('./sangyang_dorsal_left/Hand_{}.png'.format(img_id),img_org) 
            break

        # 'z' 누르면 프로그램 종료
        elif k == ord('z'):
            exit()
            
    cv2.destroyAllWindows()

# directory 설정, 적당한 이미지 수만큼 넣고 돌릴것 or (cv::OutOfMemoryError)
mypath = './test'

# 모든 이미지 list에 저장
image_files = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
images = np.empty(len(image_files), dtype=object)
for i in range(0, len(image_files)):
    images[i] = cv2.imread(join(mypath,image_files[i]), cv2.IMREAD_UNCHANGED)
    
for i in range(0, len(images)):
    # img = images[i]
    img = cv2.resize(images[i], dsize=(500,600))
    img_org = images[i]
    img_path = join(mypath,image_files[i])
    show_image(img, img_org, img_path)