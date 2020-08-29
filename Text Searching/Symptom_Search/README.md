# Symptom_Search

## 1. Symptom_Search.py
    사용자 호소 증상을 바탕으로 관련된 문서를 검색하는 검색 시스템
## 2. symp
    질병 검색 대상 문서
## 3. FILES.json
    문서의 길이 계산하여 dictionary형태로 저장
## 4. globalDictionary.json
    문서 내의 모든 단어들을 역 색인 구조를 활용하여 저장
## 5. wsymptom.dat
    역 색인 구조를 byte형태로 저장
## 6. Symptom_Keyword_Maintenance.ipynb
    서비스가 진화함에 따라서 환자 호소 증상은 다양해질 수 있으므로 확장성을 고려하여 키워드 확장을 위한 관리 파일
