from flask import Flask, request, Response, jsonify, render_template, redirect
import numpy as np
import os
from os.path import join
import requests
import json


app = Flask(__name__)

# API entry point
@app.route('/')
def index(label=None, tags=None):
    return render_template("index.html")



@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:

            image = request.files["image"]
            print(image)
            # prepare headers for http request
            content_type = 'image/jpg'
            headers = {'content-type': content_type}

            # send http request with image and receive response
            api_url = 'http://127.0.0.1:5000/predict'

            response = requests.post(api_url, data=image, headers=headers)

            # decode response
            prediction = json.loads(response.content)

            tags = prediction['tags']
            label = prediction['label']

            print(tags)
            print()
            print(label)

            return render_template("index.html", label=label, tags=tags)

        return redirect("/")

    return redirect("/")



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)