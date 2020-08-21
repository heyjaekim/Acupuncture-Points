from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, expose, AdminIndexView
from flask_admin.contrib.sqla import ModelView
from flask_dropzone import Dropzone
from flask_admin.contrib.fileadmin import FileAdmin
import os

from symptom_search import Search_symptom


app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/service', methods=['GET'])
def service(symptom=None):

    return render_template('index.html', symptom=symptom)


@app.route('/getsymp', methods=['POST', 'GET'])
def getsymp(temp=None):

    if request.method == 'POST':
        temp = request.form['symptom']
        pass

    elif request.method == 'GET':
        temp = request.args.get('symptom')
        a = Search_symptom(temp)
        result = a.search_engine()[0][0]
        return render_template('index.html', symptom=result)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)