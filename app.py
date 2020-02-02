from flask import Flask,request, jsonify, Response 
from flask_restful import Resource, Api, reqparse

from werkzeug.utils import secure_filename
from feat_extract import extract_feature
import code
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf
import sounddevice as sd
import queue

upload_folder = 'audio'
allowed_extensions = {'wav','pdf'}

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Voice Command</h1>"

def getFeature():
    return "feature"

@app.route('/uploadaudio', methods=['POST'])
def getFileAudio():
    if request.method == 'POST':
        
        f = request.files['file']
        filepath = './audio/'+secure_filename(f.filename)
        
        f.save(filepath)
        print('done')
        # mfccs,chroma,mel,contrast,tonnetz = extract_feature(f)
        
    return "done"

app.add_url_rule('/feature','feature',getFeature)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port="5000", debug=True)