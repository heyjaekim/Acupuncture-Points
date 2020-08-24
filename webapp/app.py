from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, expose, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask_dropzone import Dropzone
from flask_admin.contrib.fileadmin import FileAdmin

from speech2text import csr
import os
import math

from symptom_search import Search_symptom
from HospitalGeo.Hospital_Geo import Nearest_Hospital
from speech2text import csr

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
# simple_geoip = SimpleGeoIP(app)

###########################################################

# Flask and Flask-SQLAlchemy initialization here

admin = Admin(name='test')

admin.init_app(app)
# admin.add_view(ModelView(User, db.session))
admin.add_view(FileAdmin(upload_dir, name='Uploads'))

###########################################################


@app.route('/', methods=['GET', 'POST'])
def home():
    data = request.get_json()

    if data != None:
        lng = data['lng']
        lat = data['lat']
        # print(lng, lat)

        # need to add feature to get the current position
        nh = Nearest_Hospital(lng, lat)
        # nh = Nearest_Hospital(127.08133592498548, 37.64928136787053)
        folium_map = nh.result_map()
        folium_map.save('templates/map.html')

    return render_template('home.html')


@app.route('/service', methods=['GET', 'POST'])
def service():

    if request.method == "POST":
        f = request.files['audio_data']
        with open('voice.wav', 'wb') as voice:
            f.save(voice)
        print('file uploaded successfully')
        result = csr('./voice.wav')
        print(result)

        return render_template('service.html', request="POST", symptom=None)
    else:
        return render_template('service.html', symptom=None)



@app.route('/getsymp', methods=['POST', 'GET'])
def getsymp(symptom=None):

    if request.method == 'POST':
        # 파일 올릴때만 request.files 를 써야한다고 함
        # f = request.files.get('file')
        # symptom = request.args.get('symptom')
        # result = request.args.get('result')
        # f.save(os.path.join(upload_dir, f.filename))
        # return render_template('service.html', symptom=symptom, result=result)
        pass

    elif request.method == 'GET':

        symptom = request.args.get('symptom')
        a = Search_symptom(symptom)
        result = a.search()[2]

        return render_template('service.html', symptom=symptom, result=result)


@app.route('/upload_photo', methods=['GET', 'POST'])
def upload_photo(symptom=None, result=None):
    if request.method == 'POST':
        # 파일 올릴때만 request.files 를 써야한다고 함
        f = request.files.get('file')
        f.save(os.path.join(upload_dir, f.filename))
        symptom = request.args.get('symptom')
        result = request.args.get('result')
    return render_template('service.html', symptom=symptom, result=result)


@app.route('/map')
def openmap():
    return render_template('map.html')


if __name__ == '__main__':
    app.run()