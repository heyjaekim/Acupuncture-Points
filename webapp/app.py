from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, expose, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask_dropzone import Dropzone
from flask_admin.contrib.fileadmin import FileAdmin
import os


app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET'])
def index(symptom=None):

    return render_template('home-classic-one-page.html', symptom=symptom)


@app.route('/getsymp', methods=['POST', 'GET'])
def getsymp(temp=None):

    if request.method == 'POST':
        temp = request.form['symptom']
        pass

    elif request.method == 'GET':
        temp = request.args.get('symptom')
        return render_template('home-classic-one-page.html', symptom=temp)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)