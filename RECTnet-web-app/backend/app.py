import cv2
import io
import json
import requests

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from final_model_api import run_api
import numpy as np


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
def main():
    return jsonify({'classsfds_id': 'hiii'})

# @cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file'].read()
        np_img = np.fromstring(file, np.uint8) # convert to numpy array
        img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED) # convert numpy array to image
        result = run_api(img) # run image through our own model
        # return jsonify(str(result))
        return json.dumps(str(result))

    return '''
    <!doctype html>
    <title>Upload new image</title>
    <h1>This URI is used for uploading images</h1>
    '''

if __name__ == '__main__':
    app.run(debug=True)

# $ export FLASK_APP=hello.py
# $ python -m flask run
