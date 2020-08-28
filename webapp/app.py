import sqlite3

from flask import Flask, jsonify, request, render_template, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_dropzone import Dropzone
from flask_admin.contrib.fileadmin import FileAdmin
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import (create_access_token)


from os import path
from threading import Thread
from datetime import datetime

from Text_Searching.Symptom_Search.search_symptom import Search_symptom
from Text_Searching.Symptom_Matching.Matching_Symptom import KMT
from Text_Searching.speech2text import csr
from HospitalGeo.Hospital_Geo import Nearest_Hospital

# ## image processing stuff ##
from CV_DL.CV_check_Utils import *
from torchvision import transforms
from PIL import Image
from io import BytesIO
import torch
import cv2


basedir = path.abspath(path.dirname(__file__))
upload_dir = path.join(basedir, 'uploads')

###########################################################

# FLASK APP CONFIGURATIONS
app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Collect_Data.db'
app.config['SECRET_KEY'] = 'jae'
app.config['UPLOADED_PATH'] = upload_dir
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'



# DB SQLAlchemy AND SETTINGS
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)

# FLASK-DROPZONE
Dropzone(app)


###########################################################

# Flask and Flask-SQLAlchemy initialization here

admin = Admin(name='User_Data')
admin.init_app(app)
admin.add_view(FileAdmin(upload_dir, name='Uploads'))
voice_symptom = ""

###########################################################

# initialize models
# baseline : resnet-34 ;
model = create_model()

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


@app.route('/user/login', methods=['GET', 'POST'])
def login():
    data = request.get_json()
    if request.method == "POST":
        # userid = request.files.get('userid')
        # password = request.files.get('password')
        # username = request.files.get('username')
        username = ""
        if data['new_userid']:
            userid = data['new_userid']
            password = data['password']
            username = data['username']

            conn = sqlite3.connect('./Collect_Data.db')
            cur = conn.cursor()
            count = cur.execute("SELECT COUNT(Patient_id) FROM Patient").fetchall()
            pid = "p" + str(count[0][0])
            print(pid, userid, password, username)

            cur.execute('''INSERT INTO Patient (Patient_id, ID, Username, Gender, Password)
                        VALUES(?,?,?,?,?)''', (pid, userid, password, "", username))
            conn.commit()
            print("Inserted")
            # "WHERE NOT EXISTS(SELECT 1 FROM Patient WHERE ID='" + userid + "')"
            return render_template('login.html', username=username)
        elif data['userid']:
            userid = data['userid']
            password = data['password']
            username = data['username']

            # conn = sqlite3.connect('./Collect_Data.db')
            # cur = conn.cursor()
            # count = cur.execute("SELECT COUNT(Patient_id) FROM Patient").fetchall()
            # pid = "p" + str(count[0][0])
            # print(pid, userid, password, username)
            #
            # cur.execute('''INSERT INTO Patient (Patient_id, ID, Username, Gender, Password)
            #                         VALUES(?,?,?,?,?)''', (pid, userid, password, "", username))
            # conn.commit()
            print("Logged in")

    else:
        # if data:
        #     userid = data['userid']
        #     password = data['password']
        #     username = data['username']
        #
        #     # conn = sqlite3.connect('./Collect_Data.db')
        #     # cur = conn.cursor()
        #     # count = cur.execute("SELECT COUNT(Patient_id) FROM Patient").fetchall()
        #     # pid = "p" + str(count[0][0])
        #     # print(pid, userid, password, username)
        #     #
        #     # cur.execute('''INSERT INTO Patient (Patient_id, ID, Username, Gender, Password)
        #     #                         VALUES(?,?,?,?,?)''', (pid, userid, password, "", username))
        #     # conn.commit()
        #     print("Logged in")
        #
        #     return render_template('service.html', symptom=None)
        # else:
        print('login GET')
        return render_template('login.html', result="Fail")
    # id = request.get_json()['id']
    # password = request.get_json()['password']
    # cur.execute("SELECT * FROM Patient WHERE ID == '" + id + "'")
    # rv = cur.fetchall()
    #
    # if bcrypt.check_password_hash(rv['password'], password):
    #     access_token = create_access_token(
    #         identity={'name': rv['name'], 'sex': rv['sex']})
    #     result = access_token
    # else:
    #     result = jsonify({"error": "Invalid username and password"})

@app.route('/user/register')
def register():
    return render_template('register.html')


@app.route('/service', methods=['GET', 'POST'])
def service():

    if request.method == "POST":
        if request.files['audio_data']:
            f = request.files['audio_data']
            # uncomment here to try tour voice
            # with open('voice.wav', 'wb') as voice:
            #     f.save(voice)
            print('file uploaded successfully')

            thread1 = Thread(target=get_voice_file)
            thread1.start()

            thread1.join()
            try:
                print(voice_symptom)
                return render_template('service.html', symptom=None)

            except TypeError:
                print("증상을 정확히 말씀해주세요")
                pass
        else:
            return render_template('service.html', symptom=None)
    else:
        return render_template('service.html', symptom=None)


@app.route('/getsymp', methods=['POST', 'GET'])
def getsymp(symptom=None):
    if request.method == "POST":
        pass

    elif request.method == 'GET':
        new_acups = set()
        new_foods = set()
        new_symps = set()
        result = ""

        a = Search_symptom()
        symptom = request.args.get('symptom')
        symp_lst = a.tokenizer2(symptom)

        for s in symp_lst:
            try:
                found_symp = a.search(s)[-1]
                acups = { _ for _ in KMT.search_Acup(found_symp)}
                foods = {(k, v, len(v), img) if v else (k, " - ", 10, img) for k,v,img in KMT.search_Food(found_symp)}
                new_acups = new_acups.union(acups) if new_acups else acups
                new_foods = new_foods.union(foods) if new_foods else foods
                new_symps.add(found_symp)
            except TypeError:
                pass

        for _ in new_symps:
            result += " · " + _ if result != "" else _

        return render_template('service.html', symptom=symptom, result=result,
                               acups=sorted(new_acups), foods=sorted(new_foods))


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo(symptom=None, result=None):
    if request.method == 'POST':
        # 파일 올릴때만 request.files 를 써야한다고 함
        f = request.files.get('file')
        f.save(path.join(upload_dir, f.filename))
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


def get_voice_file():
    csr_inst = csr('./voice.wav')
    global voice_symptom
    voice_symptom = csr_inst.convert()


@app.route('/getvoice', methods=['GET'])
def getvoice():
    if request.method == 'GET':
        new_acups = set()
        new_foods = set()
        new_symps = set()
        voice_result = ""

        a = Search_symptom()
        symp_lst = a.tokenizer2(voice_symptom)
        for s in symp_lst:
            try:
                found_symp = a.search(s)[-1]
                acups = {_ for _ in KMT.search_Acup(found_symp)}
                foods = {(k, v, len(v), img) if v else (k, " - ", 10, img) for k, v, img in KMT.search_Food(found_symp)}
                new_acups = new_acups.union(acups) if new_acups else acups
                new_foods = new_foods.union(foods) if new_foods else foods
                new_symps.add(found_symp)
            except TypeError:
                pass

        for _ in new_symps:
            voice_result += " · " + _ if voice_result != "" else _

        return render_template('service.html', symptom=voice_symptom, result=voice_result,
                               acups=sorted(new_acups), foods=sorted(new_foods))


###########################################################
# Image Processing #

@app.route('/image.png')
def dl_predict():

    # checkpoint
    checkpoint_dir = './CV_DL/hapgok0823_1759org+rot+fill+rotfill_model_best.pth.tar'
    if path.isfile(checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['state_dict'])
    # transformation
    transform = transforms.ToTensor()

    # open image
    f = "./uploads/test2.jpg"
    img1 = img2 = cv2.resize(cv2.imread(f), dsize=(256, 256))
    img2 = cv2.cvtColor(clear_background(img2), cv2.COLOR_BGR2RGB)
    img2 = transform(Image.fromarray(img2))

    # get coordinate results
    if torch.cuda.is_available():
        print('running on GPU')
        model.to('cuda')
        _, result = model(img2.unsqueeze(0).to('cuda'))
        result.cpu().detach().numpy()
        x, y = result.cpu().detach().numpy().squeeze()
    else:
        print('running on CPU')
        model.to('cpu')
        _, result = model(img2.unsqueeze(0))
        x, y = result.detach().numpy().squeeze()

    # open_cv edited new image
    coord = (x, y)
    dot_size = 2
    new_img = cv2.circle(img1, (int(x), int(y)), dot_size, (0, 0, 255), -1)
    print('Label: ', checkpoint_dir.split('/')[2].split('_')[0][:-4])
    print('Coord:', coord)
    # PIL Image
    new_img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

    file_object = BytesIO()
    new_img.save(file_object, 'PNG')
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')


if __name__ == '__main__':
    app.run(debug=True)