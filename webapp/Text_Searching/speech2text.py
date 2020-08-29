import requests

class csr:
    def __init__(self, path):
        self.path = path

    def convert(self):
        client_id = "6mkfoeeed5"
        client_secret = "Dz6nzNglBRw6hd2PXyY4jbYiQPlZtQ02tsjwT1AG"
        lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
        url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang
        data = open(self.path, 'rb')
        headers = {
            "X-NCP-APIGW-API-KEY-ID": "6mkfoeeed5",
            "X-NCP-APIGW-API-KEY": "Dz6nzNglBRw6hd2PXyY4jbYiQPlZtQ02tsjwT1AG",
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(url,  data=data, headers=headers)
        rescode = response.status_code

        if(rescode == 200):
            # print(response.text)
            return response.text.split(':')[1].strip("}")[1:-1]
        else:
            print("Error : " + response.text)

# model = csr('./voice.mp3')
# print(model.convert())