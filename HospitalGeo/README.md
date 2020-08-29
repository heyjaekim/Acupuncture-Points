# HospitalGeo

## 1. Hospital.db
    공공 데이터 포털의 전국 병‧의원 찾기 서비스를 활용하여 사용자가 접근할 수 있는 전국 모든 병 의원의 정보를 DB화

## 2. Hospital_Geo visualization_Class.py
    1) 사용자의 위치를 좌표값으로 받아와 world open API를 활용하여 주소 값(도로명 주소)로 변환
    2) 도로명 주소를 바탕으로 Hospital.db에 query를 통한 지역 내 병의원 검색  ex) 서울특별시 관악구 관악로
    3) 사용자의 거리와 병 의원 사이의 거리를 고려하여 병원의 정보를 foliumn을 통한 시각화

  
