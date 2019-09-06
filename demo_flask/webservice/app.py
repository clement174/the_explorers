from flask import Flask, request, Response, jsonify
import numpy as np

from tag_model.TagModel import TagModel
from classif_model.ClassifModel import ClassifModel


app = Flask(__name__)

# Load classifier
classif_model = ClassifModel()
# Load tag generator
tag_model = TagModel(
    model_files_folder="tag_model/model_files/",
    user_top_k=15,
    score_threshold=0.3
    )
tag_model.load()

# API entry point
@app.route('/predict', methods=['POST'])
def predict():
    """
        Get request
        extract image from request
        make a prediction
        answer back to sender
    """
    r = request
    image_path = r.data

    tags = tag_model.generate_tags(image_path)
    tags_str = ",".join(tags)

    response = { 'message': tags }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)