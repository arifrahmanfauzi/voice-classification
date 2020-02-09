from flask import Flask,request, jsonify, Response 
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

from werkzeug.utils import secure_filename
from feat_extract import extract_feature

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
    return "<h1>Voice Command</h1>"

def getFeature():
    return "feature"

@app.route('/uploadaudio', methods=['POST'])
@cross_origin()
def getFileAudio():
    if request.method == 'POST':
        
        f = request.files['file']
        print(f)
        filepath = './audio/'+secure_filename(f.filename)
        
        f.save(filepath)
        X, sample_rate = sf.read(filepath, dtype='float32')
        print('done')
        # mfccs,chroma,mel,contrast,tonnetz = extract_feature(f)
        
    return jsonify({'time':X.tolist(),'sample rate':sample_rate})

app.add_url_rule('/feature','feature',getFeature)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port="5000", debug=True)