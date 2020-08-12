## image_tagger.py
설명: 정방향 손모양 사진 위에 혈자리를 손쉽게 태깅 작업하기 위한 프로그램.\
결과: 혈자리를 태깅한 사진, 좌표정보, 혈점자리, 손모양 등을 json 파일로 저장.
    
    1. 이미지 Directory 설정 및 시작 ex) mypath = './directory'
    2. 혈점자리 입력 ex) 소충, 상양, ...
    3. 손모양 입력 ex) dorsal_left/dorsal_right/palmar_left/palmar_right

## image_transformer.py
설명: 태깅된 사진과 json파일 안의 좌표정보를 이용하여 다양한 태깅 데이터들을 만들어내는 프로그램.\
결과: 혈자리 태깅된 한 사진을 각기 다른 방법으로 transform 시키고 json파일로 추합. 

## json_to_csv.py
설명: example\
결과: example

## preprocessing_json.py
설명: example\
결과: example
