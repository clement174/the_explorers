from flask import Flask, request, Response, jsonify, render_template
import numpy as np
import os
from os.path import join

from tag_model.TagModel import TagModel
from classif_model.ClassifModel import ClassifModel


app = Flask(__name__)

# Load classifier
classif_model = ClassifModel()
# Load tag generator
tag_model = TagModel(
    user_top_k=15,
    score_threshold=0.3
    )
tag_model.load()

# API entry point
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
        Get request
        extract image from request
        make a prediction
        answer back to sender
    """
    r = request
    image = r.data

    tags, tags_scores = tag_model.generate_tags(image)
    label = classif_model.get_label(tags_scores)

    response = {
        'tags': tags,
        'label': label
     }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)