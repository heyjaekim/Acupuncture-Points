# 빅데이터 청년인재 13조 - 웹앱 서비스 (Web)
-- python 3.8, Java jdk 1.8, 그리고 Jpype를 다운받고 설치해주세요.

## 프로젝트 실행 방법
1. requirements.txt에 있는 파일들을 실행하여 환경설정 실행해주세요.
(Please install requirements.txt file on local repository terminal)

    - pip install -r requirements.txt
    - opencv-python==4.4.0, torch==1.6.0, torchvision==0.7.0, cuda10.1이 제대로 설치되지 않을시
        1. pip install opencv-python 설치
        2. "pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"
            을 입력하여 설치
        3. CUDA 10.1이 없으시다면 설치후 환경설정 하는 것을 강력 추천드립니다.

2. 아래 명시되어 있는 구글 드라이브에 있는 체크 포인트들을 다운받고 CV_DL폴더 안에 포함시켜 주세요.
(Please download tar files in this link and locate them inside CV_DL directory)
    - https://drive.google.com/drive/folders/1jMAioc724P114FfOKzNs7UW0Ij5Kcarg?usp=sharing\
      
3. 환경 설정이 성공적으로 설치되었따면 app.py 에서 flask를 실행시켜 주세요.

## Web Screen shots
    
### STEP1. Install Flask and Open Homepage
![Screenshot (10)](https://user-images.githubusercontent.com/52299657/91724925-f25f0c00-ebd8-11ea-8470-2c33ddd9423c.png) 
### STEP2. Impage Upload or Take Pictures for CV_Deep Learning
![Screenshot (13)](https://user-images.githubusercontent.com/52299657/91724972-fee36480-ebd8-11ea-9988-95fcf80d0841.png)
### STEP3. Type in Symptoms that you have
![Screenshot (16)](https://user-images.githubusercontent.com/52299657/91725229-6699af80-ebd9-11ea-80c0-2a7ef44f3fb9.png)
