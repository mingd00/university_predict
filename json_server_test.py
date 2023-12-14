from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 데이터 받을 빈 리스트 생성
        data_list = []

        # 사용자가 입력한 값 불러오기
        data_list = request.get_json(force=True)['data_list']

        # json 형식으로 return
        return jsonify(data_list)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, port=7070)