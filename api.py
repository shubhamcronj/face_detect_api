import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64

app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app, support_credentials=True)


@app.route('/detect_face', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def detect_face():
    content = request.json
    #print(content)
    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    nparr = np.fromstring(base64.b64decode(content["base64"]), np.uint8)
    test1 = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    print(test1.shape)
    gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    print('Faces found: ', len(faces))
    return jsonify({"status":"OK", "faces": len(faces)})

app.run(host='0.0.0.0', port="6666")