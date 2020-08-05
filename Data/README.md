# DB/텍스트/크롤링 데이터 관련 폴더 
- 네이버 지식인 증상 크롤링(naver-kin.py)
    - 찾고자 하는 Search Word를 가진 질문에서 Expected Answer를 가진 답변을 가져오는 크롤링
    - 결과값들을 json 파일로 저장

- 유사도 체커(Similarity-checker.py)
    - 각 주치별 QnA를 저장한 json 파일을 추합하여 기준이 되는 주치 데이터 생성
    - 생성 후 증상 컨택스트로 테스트 했을 때 어떤 주치와 가장 높은 유사도를 가지고 있는지 체크
