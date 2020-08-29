import numbers
import math
import xmltodict
import json
import requests
import os 
import numpy as np 
import folium
import sqlite3
import os
import requests
from konlpy.tag import Okt
from folium import plugins

class Nearest_Hospital:

    def __init__(self, Lon, Lat):
        self.Lon = Lon
        self.Lat = Lat
        self.okt = Okt()

    def degree2radius(self, degree):
        return degree * (math.pi / 180)

    def get_harversion_distance(self, x1, y1, x2, y2, round_decimal_digits=5):
        """
        (x1,y1), (x2,y2) distance by Harversion Formula (Km)
        """
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        assert isinstance(x1, numbers.Number) and -180 <= x1 and x1 <= 180
        assert isinstance(y1, numbers.Number) and  -90 <= y1 and y1 <=  90
        assert isinstance(x2, numbers.Number) and -180 <= x2 and x2 <= 180
        assert isinstance(y2, numbers.Number) and  -90 <= y2 and y2 <=  90

        R = 6371
        dLon = self.degree2radius(x2-x1)
        dLat = self.degree2radius(y2-y1)

        a = math.sin(dLat/2) * math.sin(dLat/2) + (math.cos(self.degree2radius(y1)) * math.cos(self.degree2radius(y2)) * math.sin(dLon / 2) * math.sin(dLon / 2))
        b = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return round(R * b, round_decimal_digits)


    def location(self):
        '''
        get customer's specifi location
        '''
        geoUrl = 'http://api.vworld.kr/req/address?service=address&request=getAddress&version=2.0&crs=epsg:4326&point=' + str(self.Lon) + ',' + str(self.Lat) + '&format=xml&type=road&zipcode=true&simple=false&key=35C83A30-300F-3C4C-9224-D8712B43726D'
        response = requests.get(geoUrl)
        xmldict = xmltodict.parse(response.text)
        geoResult = json.loads(json.dumps(xmldict))

        return geoResult['response']['result']['item']['structure']['level1'], geoResult['response']['result']['item']['structure']['level2'], geoResult['response']['result']['item']['structure']['level4L']


    def search_hosp(self, level1, level2, level4L):
        '''
        get the nearest hospital in specific location
        '''
        conn = sqlite3.connect('hospital_last.db')
        cur = conn.cursor()
        cur.execute('''
        SELECT dutyName, dutyTel1, dutyDivNam, dutyAddr, dutyMapimg,wgs84Lat, wgs84Lon 
        FROM Hospital as H
        WHERE H.Add_level1 = ''' + level1 +'''
        and H.Add_level2 = ''' + level2 +'''
        and H.Add_level3 = ''' + level4L)
        res = cur.fetchall()
        return res


    def result_map(self, symptom = None, distance = 0.5):
        '''
        get customer's location and make map
        Symptom will tokenize as nouns
        Based on tokenized nouns, it will recommend the most fit hospital for patient
        '''

        custo_loc = self.location()
        l1, l2, l4 = custo_loc[0].__repr__() , custo_loc[1].__repr__() , custo_loc[2].__repr__()
        hos_loc = self.search_hosp(l1, l2, l4)

        m = folium.Map([self.Lat, self.Lon], zoom_start = 20)
        icon_red = folium.Icon(color='red')
        folium.Marker((self.Lat, self.Lon), tooltip='My location', icon = icon_red).add_to(m)


        fg = folium.FeatureGroup(name='groups')
        m.add_child(fg)

        korean_med = plugins.FeatureGroupSubGroup(fg, '한의병원')
        m.add_child(korean_med)

        dentist = plugins.FeatureGroupSubGroup(fg, '치과')
        m.add_child(dentist)

        total = plugins.FeatureGroupSubGroup(fg, '종합병원')
        m.add_child(total)

        yoyang = plugins.FeatureGroupSubGroup(fg, '요양병원')
        m.add_child(yoyang)

        bogun = plugins.FeatureGroupSubGroup(fg, '보건소')
        m.add_child(bogun)

        hospi = plugins.FeatureGroupSubGroup(fg, '의원')
        m.add_child(hospi)

        etc = plugins.FeatureGroupSubGroup(fg, '기타')
        m.add_child(etc)



        for i in range(len(hos_loc)):

            if self.get_harversion_distance(self.Lon, self.Lat, float(hos_loc[i][6]), float(hos_loc[i][5])) < distance:

                if hos_loc[i][2] == '기타' or hos_loc[i][2] == '기타(구급차)':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(etc)

                if hos_loc[i][2] == '병원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(hospi)

                if hos_loc[i][2] == '보건소':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(bogun)

                if hos_loc[i][2] == '요양병원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(yoyang)

                if hos_loc[i][2] == '의원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(hospi)

                if hos_loc[i][2] == '종합병원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(total)

                if hos_loc[i][2] == '치과병원' or hos_loc[i][2] == '치과의원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(dentist)
                if hos_loc[i][2] == '한방병원' or hos_loc[i][2] == '한의원':

                    folium.Marker([hos_loc[i][5], hos_loc[i][6]], tooltip = '병원명 :'\
                    +hos_loc[i][0] +'/' + '전화번호 : ' + hos_loc[i][1]).add_to(korean_med)

            folium.LayerControl(collapsed=False).add_to(m)

        return m

###############################################
NH = Nearest_Hospital(127.08133592498548, 37.64928136787053)
NH.result_map()
