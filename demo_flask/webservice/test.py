import requests
import json

def classify_image(image_path):
    """
        Take url of an image and send API a request to get the label and tags
    """

    # prepare headers for http request
    content_type = 'img_path/jpg'
    headers = {'content-type': content_type}

    # send http request with image and receive response
    api_url = 'http://127.0.0.1:5000/predict'
    response = requests.post(api_url, data=image_path, headers=headers)

    # decode response
    prediction = json.loads(response.content)

    return prediction["message"]


image_path = "image_test.jpg"
pred = classify_image(image_path)

print("Class:", "Not implemented yet")
print()
for tag in pred:
    name, score = tag.split("-")
    print(name, "-", score)