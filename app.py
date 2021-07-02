from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from detect_app import detect_app
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
app = Flask(__name__)

tf.app.flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
tf.app.flags.DEFINE_string('weights', './checkpoints/custom-416','path to weights file')
tf.app.flags.DEFINE_integer('size', 416, 'resize images to')
tf.app.flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
tf.app.flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
tf.app.flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
tf.app.flags.DEFINE_string('output', './detections/', 'path to output folder')
tf.app.flags.DEFINE_float('iou', 0.45, 'iou threshold')
tf.app.flags.DEFINE_float('score', 0.25, 'score threshold')
tf.app.flags.DEFINE_boolean('dont_show', False, 'dont show image output')
FLAGS = tf.app.flags.FLAGS
############################################## THE REAL DEAL ###############################################
@app.route('/detectObject' , methods=['POST'])
def mask_image():
    # print(request.files , file=sys.stderr)
    file = request.files['image'].read() ## byte file
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = detect_app(img,FLAGS)
    img = Image.fromarray(img.astype("uint8"))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img.astype("uint8"))
    # img.show()
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status':str(img_base64)})

##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

@app.route('/test' , methods=['GET','POST'])
def test():
    print("log: got at test" , file=sys.stderr)
    return jsonify({'status':'succces'})

@app.route('/')
def home():
    return render_template('./index.html')

    
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug = True)