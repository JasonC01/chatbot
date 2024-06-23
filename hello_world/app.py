import json
from flask import Flask, jsonify, request
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message='Hello from Flask on AWS Lambda!')

@app.route("/verify", methods=['POST'])
def verify():
    print(request.files['image'])

    # img = request.files['image']
    # img = Image.open(img, "rb")
    # vision_model = genai.GenerativeModel('gemini-pro-vision')
    # response = vision_model.generate_content(
    #     ["What is the string with full match to regex [A\d\d\d\d\d\d\dA-Z]", img])
    # return response.text
    return jsonify(message='Hello from Flask on AWS Lambda!')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=100)

# import requests


