from flask import Flask, request, Response, jsonify, render_template, redirect
from werkzeug.utils import secure_filename
import numpy as np
import os
from os.path import join
import requests
import json
import copy


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/images"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# API entry point
@app.route('/')
def index(label=None, tags=None, print_image=None):
    return render_template("index.html")



@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:

            # get image from form and save in static/images
            image_f = request.files["image"]
            image_name = secure_filename(image_f.filename)
            image_f.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

            # save image
            image = open(os.path.join(app.config['UPLOAD_FOLDER'], image_name), 'rb').read()

            # prepare headers for http request
            content_type = 'image/jpg'
            headers = {'content-type': content_type}

            # send http request with image and receive response
            api_url = 'http://127.0.0.1:5000/predict'
            response = requests.post(api_url, data=image, headers=headers)

            # decode response
            prediction = json.loads(response.content)

            tags  = prediction['tags']
            label = prediction['label']

            image_filename = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
            return render_template("index.html", label=label, tags=tags, print_image=image_filename)

        return redirect("/")

    return redirect("/")



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)