## image_tagger.py
설명: 정방향 손모양 사진 위에 혈자리를 손쉽게 태깅 작업하기 위한 프로그램.\
결과: 혈자리를 태깅한 사진, 좌표정보, 혈점자리, 손모양 등을 json 파일로 저장.
    
    1. 이미지 Directory 설정 및 시작 ex) mypath = './directory'
    2. 혈점자리 입력 ex) 소충, 상양, ...
    3. 손모양 입력 ex) dorsal_left/dorsal_right/palmar_left/palmar_right

## image_augmentation.py
설명: 태깅된 사진과 json파일 안의 좌표정보를 이용하여 다양한 태깅 데이터들을 만들어내는 프로그램.\
결과: 혈자리 태깅된 한 사진을 각기 다른 방법으로 transform 시키고 json파일로 추합. 

## json_to_csv.py
미사용 예정

## extract_intersection_json.py
json 파일(들)에서 좌표가 입력되지 않은 부분을 제거한 json과, 원본 이미지(png) 파일 리스트, 태깅된 이미지(png) 파일 리스트, 이 3가지 중 교집합인 부분만 추출하여 혈위이름_intersection.json으로 저장하는 프로그램\
사용 방법은 맨 마지막 줄에 적혀 있으며, directory 구조는 다음과 같은 상태여야 한다. 프로그램 내용은 다음과 같다.
<img width="539" alt="image" src="https://user-images.githubusercontent.com/63584973/90411488-d6db0800-e0e6-11ea-8269-a4c7f826f453.png">


## error_control.py
설명 : 태깅된 이미지와 원본 이미지 사이의 교집합인 부분을 체크하며, 이미지 내의 변화를 감지하여 이미지의 유사도를 추정하여 원본 이미지와 태깅된 이미지의 유사도 측정하여 왜곡을 탐지, 온전한 json 파일을 만들어 주는 역할 <br>
결과 : 교집합의 이미지, 유사도 측정하여 왜곡 탐지한 결과, json 파일 
