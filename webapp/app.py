from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_dropzone import Dropzone
from flask_admin.contrib.fileadmin import FileAdmin

import os

from Text_Searching.Symptom_Search.search_symptom import Search_symptom
from Text_Searching.Symptom_Matching.Matching_Symptom import KMT
from HospitalGeo.Hospital_Geo import Nearest_Hospital
from Text_Searching.speech2text import csr

# ## image processing stuff ##
# from CV_DL.CV_check_Utils import *
# from torchvision import transforms
# from PIL import Image
# import torch
# import cv2


basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(basedir, 'uploads')

###########################################################

app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = 'jae'
app.config['UPLOADED_PATH'] = upload_dir
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'

###########################################################

# db SQLAlchemy
db = SQLAlchemy(app)
Dropzone(app)

###########################################################

# Flask and Flask-SQLAlchemy initialization here

admin = Admin(name='test')

admin.init_app(app)
admin.add_view(FileAdmin(upload_dir, name='Uploads'))

###########################################################

# initialize models
# baseline : resnet-34 ;
# model = create_model()

###########################################################


@app.route('/', methods=['GET', 'POST'])
def home():
    data = request.get_json()

    if data != None:
        lng = data['lng']
        lat = data['lat']
        print(lng, lat)

        # need to add feature to get the current position
        nh = Nearest_Hospital(lng, lat)
        # nh = Nearest_Hospital(127.00441798640304, 37.53384172231443)
        folium_map = nh.result_map()
        folium_map.save('templates/map.html')

    return render_template('home.html')


@app.route('/service', methods=['GET', 'POST'])
def service():
    global voice_symptom
    global voice_result
    global voice_first_symp
    global voice_second_symp

    if request.method == "POST":
        f = request.files['audio_data']
        # with open('voice.wav', 'wb') as voice:
        #     f.save(voice)
        print('file uploaded successfully')
        csr_inst = csr('./voice.wav')
        voice_symptom = csr_inst.convert()
        print(voice_symptom)
        try:
            a = Search_symptom()
            symptom = a.spacing_kkma(request.args.get(voice_symptom))
            b = a.tokenizer2(symptom)
            voice_first_symp = a.search(b[0])[-1]
            voice_second_symp = a.search(b[1])[-1]
            voice_result = voice_first_symp + " · " + voice_second_symp
            return render_template('service.html', symptom=None)
        except TypeError:
            print("증상을 정확히 말씀해주세요")
            pass
    else:
        return render_template('service.html', symptom=None)

@app.route('/getsymp', methods=['POST', 'GET'])
def getsymp(symptom=None):
    if request.method == "POST":
        pass

    elif request.method == 'GET':
        foods=set()
        acups=set()
        result=""

        a = Search_symptom()
        symptom = request.args.get('symptom')
        symp_lst = a.tokenizer2(symptom)

        for s in symp_lst:
            try:
                found_symp = a.search(s)[-1]
                print(found_symp)
                acups = acups.update({_ for _ in KMT.search_Acup(found_symp)})
                foods = foods.update({(k,v,len(v)) for k,v in KMT.search_Food(found_symp)})
                result += " · " + found_symp if result != "" else found_symp
                print(result)
            except (TypeError and AttributeError):
                pass

        return render_template('service.html', symptom=symptom, result=result,
                               acups=acups, foods=foods)


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo(symptom=None, result=None):
    if request.method == 'POST':
        # 파일 올릴때만 request.files 를 써야한다고 함
        f = request.files.get('file')
        f.save(os.path.join(upload_dir, f.filename))
        symptom = request.args.get('symptom')
        result = request.args.get('result')
        matched_acups = KMT.search_Acup(symptom)

        return render_template('service.html', symptom=symptom, result=result,
                           acups=matched_acups, foods=None)
    else:
        print("upload photo didn't work")
        pass


@app.route('/map')
def openmap():
    return render_template('map.html')


@app.route('/getvoice', methods=['GET'])
def getvoice():
    if request.method == 'GET':

        matched_acups = {_ for _ in KMT.search_Acup(voice_first_symp)}.union(
            {_ for _ in KMT.search_Acup(voice_second_symp)})

        matched_foods = {(k, v, len(v)) for k, v in KMT.search_Food(voice_first_symp)}.union(
            {(k, v, len(v)) for k, v in KMT.search_Food(voice_second_symp)})

        return render_template('service.html', symptom=voice_symptom, result=voice_result,
                               acups=matched_acups, foods=matched_foods)




# @app.route('/CV')
# def DL_predict():
#
#     # checkpoint
#     checkpoint_dir = './CV_DL/hapgok0823_1759org+rot+fill+rotfill_model_best.pth.tar'
#     if os.path.isfile(checkpoint_dir):
#         checkpoint = torch.load(checkpoint_dir)
#         model.load_state_dict(checkpoint['state_dict'])
#     # transformation
#     transform = transforms.ToTensor()
#
#     # open image
#     img1 = img2 = cv2.resize(cv2.imread('./uploads/test2.jpg'), dsize=(256, 256))
#     img2 = cv2.cvtColor(clear_background(img2), cv2.COLOR_BGR2RGB)
#     img2 = transform(Image.fromarray(img2))
#
#     # get coordinate results
#     if torch.cuda.is_available():
#         print('running on GPU')
#         model.to('cuda')
#         _, result = model(img2.unsqueeze(0).to('cuda'))
#         result.cpu().detach().numpy()
#         x, y = result.cpu().detach().numpy().squeeze()
#     else:
#         print('running on CPU')
#         model.to('cpu')
#         _, result = model(img2.unsqueeze(0))
#         x, y = result.detach().numpy().squeeze()
#
#     # open_cv editted new image
#     coord = (x, y)
#     dot_size = 2
#     new_img = cv2.circle(img1, (int(x), int(y)), dot_size, (0, 0, 255), -1)
#     print('Label: ', checkpoint_dir.split('/')[2].split('_')[0][:-4])
#     print('Coord:' ,(x, y))
#     # PIL Image
#     # Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
#     return DL_Prediction(new_img, coord)


if __name__ == '__main__':
    app.run()