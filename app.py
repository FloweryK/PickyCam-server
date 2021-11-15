from flask import Flask, request, jsonify
from server.inference import Inferencer
from PIL import Image
import numpy as np
import cv2
import io
import base64
import time

# server config
HOST = 'localhost'
PORT = 8000

# inference model
model = Inferencer()

# define server
app = Flask(__name__)

@app.route('/')
def home():
    return "hello world!"

@app.route('/upload')
def upload():
    # read file
    a = time.time()
    file = request.files['file'].read()

    # convert into cv2 image
    b = time.time()
    img = np.fromstring(file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # inference
    c = time.time()
    img = model.inference(img)

    # convert into base64 and send as response
    d = time.time()
    img = Image.fromarray(img.astype('uint8'))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    e = time.time()

    print(f'{(b-a)*1000}, {(c-b)*1000}, {(d-c)*1000}, {(e-d)*1000}')

    return jsonify({'image': str(img_base64)})


if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
