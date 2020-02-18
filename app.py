from flask import Flask,request, jsonify, Response, render_template 
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

from werkzeug.utils import secure_filename
from feat_extract import extract_feature
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioAnalysis as aa

from matplotlib.pyplot import specgram

import json
import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import queue


upload_folder = 'audio'
allowed_extensions = {'wav','pdf'}

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # return "<h1>Voice Command</h1>"
    return render_template("index.php")

def getFeature():
    return "feature"

def classify(inputFile, model_type, model_name):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [Result, P, classNames] = aT.fileClassification(inputFile, model_name,
                                                    model_type)
    print("{0:s}\t{1:s}".format("Class", "Probability"))
    for i, c in enumerate(classNames):
        print("{0:s}\t{1:.2f}".format(c, P[i]))
    print("Winner class: " + classNames[int(Result)])
    winner = classNames[int(Result)]
    return winner

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        
        f = request.files['file']
        print("file receve %s"%f)
        filepath = './audio/'+secure_filename(f.filename)
        
        f.save(filepath)
        model_path = os.getcwd()+'/model/svm5classmodel'
        # model = open(model_path)
        print("model path => %s"%model_path)
        
        winner = classify(filepath, "svm_rbf", model_path)
        
    return jsonify({'predicted_class': winner})

@app.route('/uploadaudio', methods=['POST'])
@cross_origin()
def getFileAudio():
    if request.method == 'POST':
        
        f = request.files['file']
        print(f)
        filepath = './audio/'+secure_filename(f.filename)
        
        f.save(filepath)
        X, sample_rate = sf.read(filepath, dtype='float32')
        # result = classify(filepath, "svm_rbf", "svm5classmodel")
        print('done')
        # mfccs,chroma,mel,contrast,tonnetz = extract_feature(f)
        
    return jsonify({'result': sample_rate})

app.add_url_rule('/feature','feature',getFeature)

if __name__ == '__main__':
    # context = ('../../Music/localhost.crt', '../../Music/localhost.key')
    app.run(host='127.0.0.1', port="5000", debug=True)